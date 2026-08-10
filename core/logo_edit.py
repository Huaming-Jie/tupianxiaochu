# -*- coding: utf-8 -*-
"""
core/logo_edit.py —— Logo 去除与替换
================================================================================
目标：把新 Logo 贴上去以后，看起来"本来就印在那儿"。

一个 Logo 要"长在"照片里，必须同时骗过人眼的四个通道：

1. **几何**：跟随载体表面的透视。→ 用户拾取 4 个角点，做单应性变换。
2. **光照**：继承载体表面的明暗渐变（一侧受光、一侧背光）。
   → 提取背景亮度的**低频场**，以乘性方式施加到 Logo 上。
   这一步是最容易被忽略、也最能拉开真假差距的一环。
3. **色调**：整体亮度均值/对比度与周围一致。→ LAB 空间 L 通道统计匹配。
4. **质感**：同样的失焦程度与噪点颗粒。→ 高斯 σ 与噪声 σ 匹配。

另外提供 ``blend_mode``：
* ``normal``   —— 常规覆盖，适合不透明标牌、贴纸
* ``multiply`` —— 正片叠底，适合印刷在布料/纸张上的深色 Logo（保留织物纹理）
* ``screen``   —— 滤色，适合玻璃/灯箱上的浅色 Logo
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from core.inpaint import InpaintOptions, InpaintService
from utils import (LOG, add_grain, alpha_composite, binarize_mask,
                   estimate_blur_sigma, estimate_noise_sigma,
                   match_illumination, order_quad, quad_size, quad_to_mask,
                   warp_rgba_to_quad)


@dataclass
class LogoOptions:
    """Logo 贴合参数（对应 UI 面板）。"""
    blend_mode: str = "normal"        # normal | multiply | screen
    opacity: float = 1.0              # 整体不透明度
    shading_strength: float = 0.75    # 继承背景光照渐变的强度 0~1
    illum_strength: float = 0.7       # 整体色调匹配强度 0~1
    match_texture: bool = True        # 是否匹配锐度与噪点
    edge_feather: float = 1.0         # Logo 边缘羽化像素
    keep_aspect: bool = True          # 是否保持 Logo 原始宽高比
    auto_trim: bool = True            # 自动裁掉 Logo 图四周的空白/纯色边


# --------------------------------------------------------------------------- #
# 1. Logo 素材预处理
# --------------------------------------------------------------------------- #

def prepare_logo(logo_rgba: np.ndarray, auto_trim: bool = True,
                 white_to_alpha_thr: int = 246) -> np.ndarray:
    """规范化 Logo 素材：补 alpha、去白底、裁掉空白边。

    很多用户手里的 Logo 是 JPG 白底图。若直接贴上去会带一个白方块，
    所以这里对**没有 alpha 通道**的素材做一次"近白 → 透明"的处理。
    """
    if logo_rgba.ndim == 2:
        logo_rgba = cv2.cvtColor(logo_rgba, cv2.COLOR_GRAY2RGBA)
    if logo_rgba.shape[2] == 3:
        rgb = logo_rgba
        near_white = np.all(rgb >= white_to_alpha_thr, axis=2)
        alpha = np.where(near_white, 0, 255).astype(np.uint8)
        # 抗锯齿边：对 alpha 做一次轻微模糊，避免硬边
        alpha = cv2.GaussianBlur(alpha, (0, 0), 0.6)
        logo_rgba = np.dstack([rgb, alpha])

    if auto_trim:
        a = logo_rgba[:, :, 3]
        ys, xs = np.where(a > 8)
        if len(xs) > 4:
            logo_rgba = logo_rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return logo_rgba


def quad_from_bbox(bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """矩形 → 四点四边形。"""
    x0, y0, x1, y1 = bbox
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def fit_quad_keep_aspect(quad: np.ndarray, logo_hw: Tuple[int, int]) -> np.ndarray:
    """在目标四边形内按 Logo 原始宽高比做等比内接（居中）。"""
    quad = order_quad(quad)
    tl, tr, br, bl = quad
    qw, qh = quad_size(quad)
    lh, lw = logo_hw
    if qw <= 1 or qh <= 1 or lw <= 0 or lh <= 0:
        return quad

    target_ar = lw / lh
    cur_ar = qw / qh
    if abs(target_ar - cur_ar) < 1e-3:
        return quad

    if cur_ar > target_ar:          # 四边形太宽 → 水平方向内缩
        s = target_ar / cur_ar
        m = (1 - s) / 2
        top = tr - tl
        bot = br - bl
        return np.array([tl + top * m, tr - top * m,
                         br - bot * m, bl + bot * m], dtype=np.float32)
    else:                            # 四边形太高 → 垂直方向内缩
        s = cur_ar / target_ar
        m = (1 - s) / 2
        left = bl - tl
        right = br - tr
        return np.array([tl + left * m, tr + right * m,
                         br - right * m, bl - left * m], dtype=np.float32)


# --------------------------------------------------------------------------- #
# 2. 光照场提取
# --------------------------------------------------------------------------- #

def _shading_field(bg_rgb: np.ndarray, region_mask: np.ndarray,
                   sigma_ratio: float = 0.12) -> np.ndarray:
    """提取区域内背景亮度的**低频场**，归一化到均值为 1。

    做法：对亮度通道做大尺度高斯模糊（滤掉纹理细节，只留明暗渐变），
    再除以区域平均亮度。得到的场 >1 表示这里偏亮，<1 表示偏暗。
    把它乘到 Logo 上，Logo 就自动"接住"了原表面的受光方向。
    """
    lab = cv2.cvtColor(bg_rgb, cv2.COLOR_RGB2LAB)
    l = lab[:, :, 0].astype(np.float32)
    sigma = max(3.0, min(bg_rgb.shape[:2]) * sigma_ratio)
    low = cv2.GaussianBlur(l, (0, 0), sigma)

    sel = region_mask > 0
    if sel.sum() < 16:
        return np.ones_like(l)
    mean = float(low[sel].mean()) + 1e-6
    field = low / mean
    return np.clip(field, 0.55, 1.65)


# --------------------------------------------------------------------------- #
# 3. 贴合
# --------------------------------------------------------------------------- #

def place_logo(image_rgb: np.ndarray, logo_rgba: np.ndarray,
               dst_quad: np.ndarray,
               opt: Optional[LogoOptions] = None
               ) -> Tuple[np.ndarray, np.ndarray]:
    """把 Logo 透视贴合到目标四边形，返回 (结果图, 改动掩膜)。"""
    opt = opt or LogoOptions()
    h, w = image_rgb.shape[:2]

    logo = prepare_logo(logo_rgba, opt.auto_trim)
    quad = order_quad(np.asarray(dst_quad, dtype=np.float32))
    if opt.keep_aspect:
        quad = fit_quad_keep_aspect(quad, logo.shape[:2])

    # ---- ① 几何：透视变换 ----
    warped = warp_rgba_to_quad(logo, quad, (h, w))
    alpha = warped[:, :, 3].astype(np.float32) / 255.0
    if alpha.max() <= 0:
        return image_rgb.copy(), np.zeros((h, w), np.uint8)

    region = (alpha > 0.02).astype(np.uint8) * 255
    rgb = warped[:, :, :3].astype(np.float32)

    # ---- ② 光照：继承背景低频亮度场 ----
    if opt.shading_strength > 0.01:
        field = _shading_field(image_rgb, region)
        f = 1.0 + (field - 1.0) * float(opt.shading_strength)
        rgb = np.clip(rgb * f[..., None], 0, 255)

    # ---- ③ 色调：LAB 统计匹配 ----
    if opt.illum_strength > 0.01:
        ys, xs = np.where(region > 0)
        if len(xs) > 32:
            x0, x1 = xs.min(), xs.max() + 1
            y0, y1 = ys.min(), ys.max() + 1
            bg_ref = image_rgb[y0:y1, x0:x1]
            src = np.clip(rgb, 0, 255).astype(np.uint8)
            matched = match_illumination(src[y0:y1, x0:x1], bg_ref,
                                         warped[y0:y1, x0:x1, 3],
                                         strength=opt.illum_strength)
            rgb[y0:y1, x0:x1] = matched.astype(np.float32)

    # ---- ④ 质感：锐度与噪点 ----
    if opt.match_texture:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        bs = estimate_blur_sigma(gray)
        ns = estimate_noise_sigma(gray)
        if bs > 0.05:
            rgb = cv2.GaussianBlur(rgb, (0, 0), bs)
            alpha = cv2.GaussianBlur(alpha, (0, 0), bs)
        if ns > 0.3:
            rgb = add_grain(np.clip(rgb, 0, 255).astype(np.uint8), ns, seed=11
                            ).astype(np.float32)

    # ---- 边缘羽化 ----
    if opt.edge_feather > 0.05:
        alpha = cv2.GaussianBlur(alpha, (0, 0), opt.edge_feather)
    alpha = np.clip(alpha * float(opt.opacity), 0.0, 1.0)

    # ---- 混合模式 ----
    base = image_rgb.astype(np.float32)
    fg = np.clip(rgb, 0, 255)
    if opt.blend_mode == "multiply":
        blended = base * fg / 255.0
    elif opt.blend_mode == "screen":
        blended = 255.0 - (255.0 - base) * (255.0 - fg) / 255.0
    else:
        blended = fg

    a = alpha[..., None]
    out = np.clip(blended * a + base * (1 - a) + 0.5, 0, 255).astype(np.uint8)
    changed = (alpha > 0.01).astype(np.uint8) * 255
    return out, changed


# --------------------------------------------------------------------------- #
# 4. 一站式：去旧 Logo + 贴新 Logo
# --------------------------------------------------------------------------- #

def replace_logo(image_rgb: np.ndarray,
                 old_mask: Optional[np.ndarray],
                 logo_rgba: Optional[np.ndarray],
                 dst_quad: Optional[np.ndarray],
                 service: InpaintService,
                 logo_opt: Optional[LogoOptions] = None,
                 inpaint_opt: Optional[InpaintOptions] = None) -> np.ndarray:
    """完整的 Logo 替换流程。

    * 只给 ``old_mask``            → 仅去除旧 Logo
    * 只给 ``logo_rgba+dst_quad``  → 仅贴新 Logo
    * 两者都给                     → 先擦后贴（推荐）
    """
    out = image_rgb

    if old_mask is not None and binarize_mask(old_mask).max() > 0:
        o = inpaint_opt or InpaintOptions()
        o.dilate = max(o.dilate, 5)      # Logo 常带阴影/描边，膨胀大一些
        LOG.info("去除旧 Logo…")
        out = service.inpaint(out, old_mask, o)

    if logo_rgba is not None and dst_quad is not None:
        LOG.info("贴合新 Logo…")
        out, _ = place_logo(out, logo_rgba, dst_quad, logo_opt)

    return out
