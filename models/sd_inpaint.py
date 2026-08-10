# -*- coding: utf-8 -*-
"""
models/sd_inpaint.py —— Stable Diffusion Inpainting（可选增强）
================================================================================
定位
----
LaMa 擅长"**移除**"：它不会凭空造出新物体，只会把背景纹理续写过来，
所以擦除水印/文字/杂物又快又稳。

但当被擦除区域**面积巨大**、或需要"**生成**"出合理的新内容（例如整块
招牌被挖掉后需要补出墙面结构）时，LaMa 会产生糊状区域。此时可切换到
Stable Diffusion Inpainting，用文本提示引导生成。

代价
----
* 权重 ~2.5 GB（fp16），显存需求 ≥ 6 GB
* 单次推理数秒到数十秒
* **有 GPU 时才建议启用**，CPU 上一张图可能要几分钟

因此本模块是 **懒加载** 的：只有用户在界面上主动勾选"高质量生成式修复"
时才会去下载和加载权重。
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from utils import LOG

#: 默认模型仓库（HF 上的标准 inpainting 模型）
DEFAULT_REPO = "stabilityai/stable-diffusion-2-inpainting"
FALLBACK_REPO = "runwayml/stable-diffusion-inpainting"


class SDInpainter:
    """Stable Diffusion Inpainting 封装（懒加载）。"""

    def __init__(self, repo: str = DEFAULT_REPO, device: str = "auto"):
        self.repo = repo
        self._pipe = None
        self.device = device
        self.available = False

    # ------------------------------------------------------------------ #
    @staticmethod
    def probe() -> tuple[bool, str]:
        """探测环境是否具备运行条件，返回 (是否可用, 说明)。"""
        try:
            import torch
        except ImportError:
            return False, "未安装 PyTorch"
        except OSError as e:
            # Windows 上常见 c10.dll 初始化失败（缺少 VC++ 运行时等）
            return False, f"PyTorch DLL 加载失败 ({e})"
        try:
            import diffusers  # noqa: F401
        except ImportError:
            return False, "未安装 diffusers（pip install diffusers transformers accelerate）"
        if not torch.cuda.is_available():
            return True, "无 CUDA，可运行但极慢（不推荐）"
        free = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if free < 5.5:
            return True, f"显存仅 {free:.1f}GB，可能 OOM"
        return True, f"CUDA 可用，显存 {free:.1f}GB"

    # ------------------------------------------------------------------ #
    def load(self, progress_cb=None) -> bool:
        """加载管线。首次调用会触发权重下载（走 hf-mirror）。"""
        if self._pipe is not None:
            return True
        ok, msg = self.probe()
        if not ok:
            LOG.warning("SD Inpainting 不可用：%s", msg)
            return False

        import torch
        from diffusers import StableDiffusionInpaintPipeline

        dev = self.device
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if dev == "cuda" else torch.float32

        for repo in (self.repo, FALLBACK_REPO):
            try:
                if progress_cb:
                    progress_cb(0, 0, f"加载 {repo}（首次需下载数 GB）…")
                pipe = StableDiffusionInpaintPipeline.from_pretrained(
                    repo, torch_dtype=dtype, safety_checker=None,
                    requires_safety_checker=False,
                )
                pipe = pipe.to(dev)
                if dev == "cuda":
                    pipe.enable_attention_slicing()      # 降显存
                    try:
                        pipe.enable_xformers_memory_efficient_attention()
                    except Exception:
                        pass
                self._pipe = pipe
                self.device = dev
                self.repo = repo
                self.available = True
                LOG.info("SD Inpainting 已加载：%s @ %s", repo, dev)
                return True
            except Exception as e:
                LOG.warning("加载 %s 失败：%s", repo, e)
        return False

    # ------------------------------------------------------------------ #
    def __call__(self, image_rgb: np.ndarray, mask: np.ndarray,
                 prompt: str = "clean background, seamless, photorealistic",
                 negative_prompt: str = "text, watermark, logo, letters, "
                                        "signature, blurry, distorted, artifacts",
                 steps: int = 25, guidance: float = 7.5,
                 seed: Optional[int] = None) -> np.ndarray:
        """对 ROI 执行生成式修复，返回与输入同尺寸的结果。

        SD 内部工作分辨率固定为 512 的倍数，所以这里先把 ROI 缩放到
        512×512（或 768），推理后再 **LANCZOS 放大回原尺寸**。
        由于最终仍然只把掩膜内像素贴回原图，掩膜外的清晰度不受影响。
        """
        if self._pipe is None and not self.load():
            raise RuntimeError("SD Inpainting 未能加载")

        import torch
        from PIL import Image

        h0, w0 = image_rgb.shape[:2]
        base = 512 if "2-inpainting" not in self.repo else 512
        img = cv2.resize(image_rgb, (base, base), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(mask, (base, base), interpolation=cv2.INTER_NEAREST)

        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(int(seed))

        out = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=Image.fromarray(img),
            mask_image=Image.fromarray(msk),
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=gen,
        ).images[0]

        res = np.array(out.convert("RGB"))
        return cv2.resize(res, (w0, h0), interpolation=cv2.INTER_LANCZOS4)

    def unload(self):
        """释放显存。"""
        self._pipe = None
        self.available = False
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
