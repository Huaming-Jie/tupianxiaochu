# -*- coding: utf-8 -*-
"""
core/inpaint.py —— 统一修复调度服务
================================================================================
这是整个项目的"中枢"。所有"擦除"类操作（物体、水印、文字、旧 Logo）
最终都汇聚到 ``InpaintService.inpaint()``。

流水线
------
::

    原图 + 掩膜
        ↓ ① 掩膜预处理（膨胀，吃掉抗锯齿边缘）
        ↓ ② 连通域拆分（多个互不相邻的目标分别处理，避免一个大 ROI 拖慢全图）
        ↓ ③ 计算带上下文的 ROI 并裁切
        ↓ ④ 送入引擎（LaMa / SD / OpenCV）
        ↓ ⑤ 噪点与锐度匹配（让补丁与原图"同样脏"）
        ↓ ⑥ 羽化 alpha 无损回贴
    结果图（掩膜外逐像素等于原图）

为什么要拆连通域？
------------------
若用户在一张 6000×4000 的图上零散涂了 5 个小点，直接取整体包围盒会得到
一个近乎全图的 ROI，既慢又让模型的注意力被稀释。分开处理后每个 ROI 只有
几百像素，速度与质量双赢。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import cv2
import numpy as np

from models.lama import LamaInpainter
from utils import (LOG, add_grain, binarize_mask, compute_roi, dilate_mask,
                   estimate_noise_sigma, mask_bbox, paste_back)


@dataclass
class InpaintOptions:
    """一次修复任务的可调参数（对应 UI 右侧面板）。"""
    method: str = "auto"          # auto | lama | sd | opencv
    dilate: int = 4               # 掩膜膨胀半径（像素）
    feather: int = 4              # 回贴羽化半径（像素）
    context_ratio: float = 0.7    # ROI 上下文比例
    max_side: int = 1600          # 单个 ROI 最长边上限
    match_grain: bool = True      # 是否做噪点匹配
    split_components: bool = True # 是否按连通域拆分
    sd_prompt: str = "clean background, seamless texture, photorealistic"
    sd_steps: int = 25
    sd_seed: Optional[int] = None


ProgressCb = Callable[[float, str], None]     # (0~1 进度, 文本)


class InpaintService:
    """修复服务（单例式持有各引擎，避免重复加载权重）。"""

    def __init__(self, device: str = "auto", progress_cb: Optional[ProgressCb] = None):
        self.device = device
        self._progress = progress_cb
        self.lama = LamaInpainter(device=device,
                                  progress_cb=self._dl_progress)
        self._sd = None                    # 懒加载

    # ------------------------------------------------------------------ #
    def _dl_progress(self, cur: int, total: int, text: str):
        """把下载器的字节级进度转成 0~1 进度回调。"""
        if self._progress:
            p = (cur / total) if total else 0.0
            self._progress(min(0.99, p), text)

    def _report(self, p: float, text: str):
        if self._progress:
            self._progress(p, text)

    # ------------------------------------------------------------------ #
    @property
    def sd(self):
        """懒加载 SD 引擎。"""
        if self._sd is None:
            from models.sd_inpaint import SDInpainter
            self._sd = SDInpainter(device=self.device)
        return self._sd

    def warmup(self) -> str:
        """预热主引擎（在后台线程调用），返回状态描述。"""
        self.lama.load()
        return self.lama.describe()

    # ------------------------------------------------------------------ #
    def inpaint(self, image_rgb: np.ndarray, mask: np.ndarray,
                opt: Optional[InpaintOptions] = None) -> np.ndarray:
        """核心入口：按掩膜擦除并智能填充。

        参数
        ----
        image_rgb : (H,W,3) uint8
        mask      : (H,W) uint8，非零 = 待擦除
        opt       : 参数对象

        返回
        ----
        新图（原图不被修改）
        """
        opt = opt or InpaintOptions()
        mask = binarize_mask(mask)
        if mask.max() == 0:
            LOG.info("掩膜为空，跳过修复")
            return image_rgb.copy()

        work_mask = dilate_mask(mask, opt.dilate)
        result = image_rgb.copy()

        regions = self._split_regions(work_mask) if opt.split_components else [work_mask]
        total = max(1, len(regions))
        LOG.info("修复任务：%d 个区域，方法=%s", total, opt.method)

        for i, sub in enumerate(regions):
            self._report(i / total, f"修复区域 {i + 1}/{total}…")
            result = self._inpaint_single(result, sub, opt)

        self._report(1.0, "修复完成")
        return result

    # ------------------------------------------------------------------ #
    @staticmethod
    def _split_regions(mask: np.ndarray, max_regions: int = 24) -> List[np.ndarray]:
        """按连通域把掩膜拆成若干独立子掩膜。

        若连通域过多（用户大面积涂抹），退化为整体处理以免任务数爆炸。
        """
        n, labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
        if n <= 2 or n - 1 > max_regions:
            return [mask]
        out = []
        for k in range(1, n):
            sub = np.zeros_like(mask)
            sub[labels == k] = 255
            if sub.sum() > 0:
                out.append(sub)
        return out or [mask]

    # ------------------------------------------------------------------ #
    def _inpaint_single(self, image_rgb: np.ndarray, mask: np.ndarray,
                        opt: InpaintOptions) -> np.ndarray:
        """处理单个连通区域。"""
        roi = compute_roi(mask, opt.context_ratio, min_pad=48, max_side=opt.max_side)
        if roi is None:
            return image_rgb

        x0, y0, x1, y1 = roi.as_tuple()
        img_roi = image_rgb[y0:y1, x0:x1]
        msk_roi = mask[y0:y1, x0:x1]

        method = self._resolve_method(opt.method, msk_roi)

        if method == "sd":
            try:
                patch = self.sd(img_roi, msk_roi, prompt=opt.sd_prompt,
                                steps=opt.sd_steps, seed=opt.sd_seed)
            except Exception as e:
                LOG.warning("SD 修复失败，回退 LaMa：%s", e)
                patch = self.lama(img_roi, msk_roi)
        elif method == "opencv":
            patch = LamaInpainter._run_opencv(img_roi, msk_roi)
        else:
            patch = self.lama(img_roi, msk_roi)

        if opt.match_grain:
            patch = self._match_grain(img_roi, patch, msk_roi)

        return paste_back(image_rgb, patch, roi, mask, opt.feather)

    # ------------------------------------------------------------------ #
    def _resolve_method(self, method: str, mask_roi: np.ndarray) -> str:
        """auto 模式的决策逻辑。

        规则很朴素但很有效：
        * 掩膜占 ROI 面积 **> 45%** → 需要"生成"而非"续写"，若 SD 可用则用 SD
        * 否则一律 LaMa（更快、更不容易乱画东西）
        """
        if method != "auto":
            return method
        ratio = float((mask_roi > 0).sum()) / float(mask_roi.size + 1e-6)
        if ratio > 0.45:
            from models.sd_inpaint import SDInpainter
            ok, _ = SDInpainter.probe()
            try:
                import torch
                if ok and torch.cuda.is_available():
                    LOG.info("掩膜占比 %.0f%%，切换到 SD 生成式修复", ratio * 100)
                    return "sd"
            except Exception:
                pass
        return "lama"

    # ------------------------------------------------------------------ #
    @staticmethod
    def _match_grain(orig_roi: np.ndarray, patch: np.ndarray,
                     mask_roi: np.ndarray) -> np.ndarray:
        """让修复补丁具备与周边一致的噪声颗粒。

        深度模型的输出通常"过于干净"——在一张有胶片颗粒或 JPEG 噪声的照片上，
        一块光滑无噪的区域即使颜色完全正确，人眼依然能察觉。
        这里测量掩膜**外围**的噪声强度，只在掩膜**内部**补上等量噪声。
        """
        try:
            gray = cv2.cvtColor(orig_roi, cv2.COLOR_RGB2GRAY)
            outside = (mask_roi == 0)
            if outside.sum() < 64:
                return patch
            sigma = estimate_noise_sigma(gray)
            if sigma < 0.4:
                return patch
            noisy = add_grain(patch, sigma, seed=1234)
            m = (mask_roi > 0)[..., None]
            return np.where(m, noisy, patch)
        except Exception:
            return patch

    # ------------------------------------------------------------------ #
    def describe(self) -> str:
        """引擎状态描述（UI 状态栏用）。"""
        return self.lama.describe()
