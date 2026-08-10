# -*- coding: utf-8 -*-
"""
models/lama.py —— LaMa 图像修复引擎（三级降级）
================================================================================
LaMa（Large Mask Inpainting, WACV 2022）是目前"擦除物体/水印/文字"这类
**纯移除类任务** 性价比最高的模型：

* 快速傅里叶卷积（FFC）让感受野在浅层就覆盖全图，因此对大面积掩膜、
  重复纹理（砖墙、栅栏、草地）的续写能力远超传统 PatchMatch；
* 全卷积结构，**输入尺寸任意**（只要能被 8 整除），无需切图；
* 权重仅 ~200 MB，纯 CPU 上 512×512 约 1~3 秒，符合"自适应无 GPU"的要求。

三级降级策略
------------
1. **TorchScript**（首选）：``big-lama.pt``，动态尺寸，CPU/GPU 通吃
2. **ONNX Runtime**（备选）：无 PyTorch 环境时使用；若模型为固定尺寸导出，
   自动缩放到模型要求的分辨率再缩放回来
3. **OpenCV Telea/NS**（兜底）：无网络、无权重时仍保证程序可用，
   对小面积、弱纹理区域效果尚可

任何一级失败都会自动落到下一级，**绝不让主程序崩溃**。
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np

from utils import LOG, binarize_mask, pad_to_multiple, unpad
from .downloader import get_model_path


class LamaInpainter:
    """LaMa 修复引擎封装。

    用法::

        engine = LamaInpainter(device="auto")
        result = engine(image_rgb, mask_u8)     # mask: 255 表示"要擦掉"
    """

    def __init__(self, device: str = "auto", auto_download: bool = True,
                 progress_cb=None):
        self.backend: str = "opencv"          # torchscript / onnx / opencv
        self.device: str = "cpu"
        self._model = None
        self._session = None
        self._onnx_input_size: Optional[tuple] = None
        self._progress_cb = progress_cb
        self._auto_download = auto_download
        self._requested_device = device

    # ------------------------------------------------------------------ #
    # 加载
    # ------------------------------------------------------------------ #
    def load(self) -> str:
        """按优先级加载后端，返回最终生效的后端名。幂等，可重复调用。"""
        if self.backend != "opencv" and (self._model or self._session):
            return self.backend

        if self._try_torchscript():
            return self.backend
        if self._try_onnx():
            return self.backend

        LOG.warning("LaMa 权重不可用，已降级为 OpenCV 传统修复（质量较低）")
        self.backend = "opencv"
        return self.backend

    def _resolve_device(self) -> str:
        """自适应选择计算设备：有 CUDA 用 CUDA，否则 CPU。"""
        if self._requested_device != "auto":
            return self._requested_device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            # Apple Silicon
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def _try_torchscript(self) -> bool:
        """尝试加载 TorchScript 版 big-lama.pt。"""
        # 允许通过环境变量强制跳过 TorchScript（例如 torch 环境内存不足/
        # 加载即崩溃时，直接走更轻量的 ONNX 后端，避免进程被 SIGSEGV 拖死）。
        if os.environ.get("SIE_NO_TORCH") == "1":
            LOG.info("SIE_NO_TORCH=1，跳过 TorchScript 后端")
            return False
        try:
            import torch
        except ImportError:
            LOG.info("未安装 PyTorch，跳过 TorchScript 后端")
            return False
        except OSError as e:
            LOG.warning("PyTorch DLL 加载失败，跳过 TorchScript 后端：%s", e)
            return False

        path = get_model_path("lama_ts", progress=self._progress_cb,
                              auto_download=self._auto_download)
        if not path or not os.path.isfile(path):
            return False

        try:
            dev = self._resolve_device()
            model = torch.jit.load(path, map_location="cpu")
            model.eval()
            model.to(dev)
            self._model = model
            self.device = dev
            self.backend = "torchscript"
            LOG.info("LaMa(TorchScript) 已加载，设备=%s", dev)
            return True
        except Exception as e:
            LOG.warning("TorchScript 加载失败：%s", e)
            return False

    def _try_onnx(self) -> bool:
        """尝试加载 ONNX 版（无 PyTorch 时的备选）。"""
        try:
            import onnxruntime as ort
        except ImportError:
            return False

        path = get_model_path("lama_onnx", progress=self._progress_cb,
                              auto_download=self._auto_download)
        if not path or not os.path.isfile(path):
            return False

        try:
            providers = ["CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers.insert(0, "CUDAExecutionProvider")
            sess = ort.InferenceSession(path, providers=providers)

            # 记录模型是否为固定输入尺寸
            shape = sess.get_inputs()[0].shape          # 形如 [1,3,H,W]
            if isinstance(shape[2], int) and isinstance(shape[3], int):
                self._onnx_input_size = (int(shape[3]), int(shape[2]))  # (w,h)
            self._session = sess
            self.backend = "onnx"
            self.device = "cuda" if providers[0].startswith("CUDA") else "cpu"
            LOG.info("LaMa(ONNX) 已加载，固定尺寸=%s", self._onnx_input_size)
            return True
        except Exception as e:
            LOG.warning("ONNX 加载失败：%s", e)
            return False

    # ------------------------------------------------------------------ #
    # 推理
    # ------------------------------------------------------------------ #
    def __call__(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """执行修复。

        参数
        ----
        image_rgb : (H,W,3) uint8 RGB
        mask      : (H,W) uint8，非零处表示"要被擦除并重绘"

        返回
        ----
        (H,W,3) uint8 RGB，尺寸与输入完全一致
        """
        mask = binarize_mask(mask)
        if mask.max() == 0:
            return image_rgb.copy()

        self.load()
        try:
            if self.backend == "torchscript":
                return self._run_torchscript(image_rgb, mask)
            if self.backend == "onnx":
                return self._run_onnx(image_rgb, mask)
        except Exception as e:
            LOG.error("LaMa 推理异常，降级到 OpenCV：%s", e)
        return self._run_opencv(image_rgb, mask)

    # -------------------------- 各后端实现 ---------------------------- #
    def _run_torchscript(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        import torch

        img_p, hw = pad_to_multiple(image_rgb, 8)
        msk_p, _ = pad_to_multiple(mask, 8)

        # NCHW, float32, 0~1
        x = torch.from_numpy(
            img_p.transpose(2, 0, 1).astype(np.float32) / 255.0).unsqueeze(0)
        m = torch.from_numpy(
            (msk_p > 127).astype(np.float32)).unsqueeze(0).unsqueeze(0)

        x, m = x.to(self.device), m.to(self.device)
        with torch.no_grad():
            y = self._model(x, m)

        out = y[0].permute(1, 2, 0).detach().cpu().numpy()
        # 兼容两种导出：输出 0~1 或 0~255
        if float(np.nanmax(out)) <= 1.5:
            out = out * 255.0
        out = np.clip(out, 0, 255).astype(np.uint8)
        return unpad(out, hw)

    def _run_onnx(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        h0, w0 = image_rgb.shape[:2]

        if self._onnx_input_size:           # 固定尺寸导出：先缩放
            tw, th = self._onnx_input_size
            img = cv2.resize(image_rgb, (tw, th), interpolation=cv2.INTER_AREA)
            msk = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)
            pad_hw = (th, tw)
        else:
            img, pad_hw = pad_to_multiple(image_rgb, 8)
            msk, _ = pad_to_multiple(mask, 8)

        x = img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        m = ((msk > 127).astype(np.float32))[None, None]

        names = [i.name for i in self._session.get_inputs()]
        out = self._session.run(None, {names[0]: x, names[1]: m})[0]
        out = out[0].transpose(1, 2, 0)
        if float(np.nanmax(out)) <= 1.5:
            out = out * 255.0
        out = np.clip(out, 0, 255).astype(np.uint8)

        if self._onnx_input_size:
            out = cv2.resize(out, (w0, h0), interpolation=cv2.INTER_LANCZOS4)
        else:
            out = unpad(out, (h0, w0))
        return out

    @staticmethod
    def _run_opencv(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """OpenCV 兜底：Telea 与 Navier-Stokes 双算法融合。

        单独用任一算法都容易出现"糊块"；两者取平均能略微改善纹理连续性。
        这只是保底手段，质量与 LaMa 不在一个量级。
        """
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        r = max(3, int(min(image_rgb.shape[:2]) * 0.01))
        a = cv2.inpaint(bgr, mask, r, cv2.INPAINT_TELEA)
        b = cv2.inpaint(bgr, mask, r, cv2.INPAINT_NS)
        merged = cv2.addWeighted(a, 0.5, b, 0.5, 0)
        return cv2.cvtColor(merged, cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------------------ #
    def describe(self) -> str:
        """返回当前引擎状态的可读描述（用于 UI 状态栏）。"""
        names = {"torchscript": "LaMa · TorchScript",
                 "onnx": "LaMa · ONNX",
                 "opencv": "OpenCV 传统修复（未加载 LaMa）"}
        return f"{names.get(self.backend, self.backend)}  [{self.device}]"
