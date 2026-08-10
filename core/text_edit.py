# -*- coding: utf-8 -*-
"""
core/text_edit.py —— 文字消除与文字修改（风格保持）
================================================================================
本模块解决两件事：

A. **文字消除**：把画面上的文字彻底抹掉，且不留糊块。
B. **文字修改**：把"原文字"换成"新文字"，同时保持字体观感、颜色、
   字号、倾斜角、透视变形与光照质感。

关于"风格保持"的工程思路
------------------------
理想方案是 SRNet / AnyText 这类专门的场景文字编辑模型——它们能直接学到
笔画纹理与背景耦合关系。但它们的现实代价是：权重大、显存高、中文字形
可控性差、对小字号退化明显。

因此这里采用一条 **可解释、可调、CPU 可跑** 的参数化路径，
把"风格"拆解成七个可测量的量，逐个从原图里估计出来：

===================  ====================================================
风格维度              估计方法
===================  ====================================================
① 前景色（笔画色）     ROI 内 2-means 聚类，取像素数较少的一类
② 背景色              2-means 的多数类（用于判断深底浅字还是浅底深字）
③ 字号                笔画掩膜的实际墨迹高度（不是框高，框通常有内边距）
④ 粗细（是否加粗）     笔画面积 / 墨迹外接框面积 → 笔画占空比
⑤ 倾斜与旋转          OCR 四点多边形上边缘的方位角
⑥ 透视变形            直接用四点做 ``getPerspectiveTransform``，天然携带透视
⑦ 纹理质感            匹配周边的锐度（高斯 σ）与噪声（MAD σ），并叠加回新字
===================  ====================================================

估计完成后：**擦除原字 → 渲染新字（4× 超采样）→ 透视变形到原四边形 →
光照/锐度/噪点匹配 → alpha 合成**。掩膜之外的像素完全不动。

如果环境里有 GPU + diffusers，还可以在最后追加一次低强度的扩散精修
（``harmonize=True``），让新字与背景的耦合更自然。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from core.inpaint import InpaintOptions, InpaintService
from utils import (LOG, TextBox, add_grain, alpha_composite, binarize_mask,
                   dominant_two_colors, estimate_blur_sigma,
                   estimate_noise_sigma, fit_font_size, has_cjk,
                   list_available_fonts, order_quad,
                   pick_font, quad_angle, quad_size, quad_to_mask,
                   render_text_rgba, warp_rgba_to_quad)


# --------------------------------------------------------------------------- #
# 风格描述
# --------------------------------------------------------------------------- #

@dataclass
class TextStyle:
    """从原图中估计出的文字风格参数。"""
    fg_color: Tuple[int, int, int] = (0, 0, 0)      # ① 笔画颜色
    bg_color: Tuple[int, int, int] = (255, 255, 255)  # ② 背景颜色
    ink_height: float = 20.0                         # ③ 墨迹高度(px)
    ink_width: float = 60.0                          #    墨迹宽度(px)
    stroke_ratio: float = 0.2                        # ④ 笔画占空比
    bold: bool = False                               # ④ 是否加粗
    angle: float = 0.0                               # ⑤ 倾斜角(度)
    quad: np.ndarray = field(default_factory=lambda: np.zeros((4, 2), np.float32))  # ⑥
    blur_sigma: float = 0.0                          # ⑦ 锐度匹配
    noise_sigma: float = 0.0                         # ⑦ 噪点匹配
    font_path: str = ""                              #    选中的字体
    letter_spacing: float = 0.0                      #    字距
    opacity: float = 1.0                             #    整体不透明度

    def describe(self) -> str:
        r, g, b = [int(v) for v in self.fg_color]
        return (f"颜色 #{r:02X}{g:02X}{b:02X}｜字高 {self.ink_height:.0f}px｜"
                f"{'粗体' if self.bold else '常规'}｜倾斜 {self.angle:.1f}°｜"
                f"字体 {os.path.basename(self.font_path) or '自动'}")


# --------------------------------------------------------------------------- #
# 1. 笔画级掩膜
# --------------------------------------------------------------------------- #

def stroke_mask_in_quad(image_rgb: np.ndarray, quad: np.ndarray,
                        dilate: int = 1) -> np.ndarray:
    """在给定四边形内提取"笔画级"文字掩膜（而不是整块矩形）。

    为什么不用整框？因为文字框内绝大部分是背景。只擦笔画，
    LaMa 需要编造的内容更少，背景纹理（木纹、渐变、照片细节）
    能被最大限度保留下来。

    做法：框内 2-means 聚类 → 每个像素归到"更接近前景色"还是"更接近背景色"。
    这比固定阈值二值化稳健得多（能处理白底黑字、黑底白字、彩底彩字）。
    """
    h, w = image_rgb.shape[:2]
    box_mask = quad_to_mask((h, w), quad)
    ys, xs = np.where(box_mask > 0)
    if len(xs) < 16:
        return box_mask

    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    roi = image_rgb[y0:y1, x0:x1]
    sub_box = box_mask[y0:y1, x0:x1]

    fg, bg, _ = dominant_two_colors(roi[sub_box > 0].reshape(-1, 1, 3))
    d_fg = np.linalg.norm(roi.astype(np.float32) - fg[None, None, :], axis=2)
    d_bg = np.linalg.norm(roi.astype(np.float32) - bg[None, None, :], axis=2)

    sm = np.zeros((h, w), np.uint8)
    local = ((d_fg < d_bg) & (sub_box > 0)).astype(np.uint8) * 255

    # 去掉孤立噪点
    local = cv2.morphologyEx(local, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    if dilate > 0:
        local = cv2.dilate(local, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate * 2 + 1, dilate * 2 + 1)))

    # 保护：若笔画占比异常（>80% 或 <1%），说明聚类失败，退回整框
    ratio = float((local > 0).sum()) / float((sub_box > 0).sum() + 1e-6)
    if ratio > 0.8 or ratio < 0.01:
        return box_mask

    sm[y0:y1, x0:x1] = local
    return sm


def build_text_mask(image_rgb: np.ndarray, quads: Sequence[np.ndarray],
                    tight: bool = True, dilate: int = 1) -> np.ndarray:
    """把多个文本四边形合成一张掩膜。

    tight=True → 笔画级（推荐，背景保留最多）
    tight=False → 整框（文字紧贴复杂背景、或笔画有描边/阴影时更保险）
    """
    h, w = image_rgb.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    for q in quads:
        m = stroke_mask_in_quad(image_rgb, q, dilate) if tight \
            else quad_to_mask((h, w), q)
        mask = cv2.bitwise_or(mask, m)
    return mask


# --------------------------------------------------------------------------- #
# 2. 文字消除
# --------------------------------------------------------------------------- #

def _upright_mask(mask: np.ndarray, angle: float) -> np.ndarray:
    """把可能倾斜的笔画掩膜摆正（按角度反向旋转）。"""
    if abs(angle) < 0.5 or mask.size == 0:
        return mask
    ys, xs = np.where(mask > 0)
    if len(xs) < 4:
        return mask
    cx, cy = float(xs.mean()), float(ys.mean())
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    return cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]),
                          flags=cv2.INTER_NEAREST, borderValue=0)


def _normalize_mask(mask: np.ndarray, n: int = 220) -> Optional[np.ndarray]:
    """裁到笔画外接框，再拉伸到固定画布（位置/缩放归一化，便于跨字体比对）。"""
    ys, xs = np.where(mask > 0)
    if len(xs) < 4:
        return None
    sub = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return cv2.resize(sub, (n, n), interpolation=cv2.INTER_NEAREST)


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a > 0, b > 0).sum())
    union = int(np.logical_or(a > 0, b > 0).sum())
    return inter / float(union + 1e-6)


def match_font_to_original(image_rgb: np.ndarray, quad: np.ndarray,
                           original_text: str, font_size: float,
                           angle: float = 0.0) -> Optional[str]:
    """在原图已知文字的前提下，于系统可用字体里挑与原字笔画最像的字体。

    做法：把原字笔画掩膜摆正、归一化；对每个候选字体渲染同一段原文、
    同样归一化，算 IoU（交并比），取最高者。这样替换后的新字会沿用
    与原图最接近的字体，而不是永远掉到通用默认字体。

    返回字体路径；若原文为空或没有任何候选能渲染（豆腐块）则返回 None，
    上层会回退到 pick_font 的默认选择。
    """
    if not original_text:
        return None
    orig = stroke_mask_in_quad(image_rgb, quad, dilate=0)
    if int((orig > 0).sum()) < 30:
        return None
    orig_u = _upright_mask(orig, -angle)
    orig_n = _normalize_mask(orig_u)
    if orig_n is None:
        return None

    cands = list_available_fonts(cjk=has_cjk(original_text))
    if not cands:
        return None

    best_fp, best_iou = None, -1.0
    for fp in cands:
        try:
            rgba = render_text_rgba(original_text, fp, int(round(font_size)),
                                    (0, 0, 0), supersample=1)
        except Exception:
            continue
        if rgba.size == 0:
            continue
        cand_mask = (rgba[:, :, 3] > 0).astype(np.uint8)
        if int(cand_mask.sum()) < 5:
            continue  # 该字体不含此字形（豆腐块），跳过
        cand_n = _normalize_mask(cand_mask)
        if cand_n is None:
            continue
        iou = _mask_iou(orig_n, cand_n)
        if iou > best_iou:
            best_iou, best_fp = iou, fp
    return best_fp


def erase_text(image_rgb: np.ndarray, quads: Sequence[np.ndarray],
               service: InpaintService, tight: bool = True,
               opt: Optional[InpaintOptions] = None) -> np.ndarray:
    """擦除指定四边形内的文字。"""
    if not len(quads):
        return image_rgb.copy()
    mask = build_text_mask(image_rgb, quads, tight=tight, dilate=1)
    o = opt or InpaintOptions()
    # 文字笔画细，膨胀要略大一点，避免残留描边灰边
    o.dilate = max(o.dilate, 3 if tight else 2)
    return service.inpaint(image_rgb, mask, o)


# --------------------------------------------------------------------------- #
# 3. 风格估计
# --------------------------------------------------------------------------- #

def estimate_style(image_rgb: np.ndarray, quad: np.ndarray,
                   font_hint: Optional[str] = None,
                   sample_text: str = "") -> TextStyle:
    """从原图的文字区域反推出 TextStyle。"""
    h, w = image_rgb.shape[:2]
    quad = order_quad(quad)
    st = TextStyle(quad=quad.astype(np.float32))

    box = quad_to_mask((h, w), quad)
    ys, xs = np.where(box > 0)
    if len(xs) < 16:
        return st
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    roi = image_rgb[y0:y1, x0:x1]

    # ① ② 前景色 / 背景色
    sub = box[y0:y1, x0:x1]
    fg, bg, ratio = dominant_two_colors(roi[sub > 0].reshape(-1, 1, 3))
    st.fg_color = tuple(int(round(float(v))) for v in fg)
    st.bg_color = tuple(int(round(float(v))) for v in bg)
    st.stroke_ratio = float(ratio)

    # ③ 墨迹尺寸：用笔画掩膜的真实外接框，而不是 OCR 框（后者含内边距）
    sm = stroke_mask_in_quad(image_rgb, quad, dilate=0)
    sys_, sxs_ = np.where(sm > 0)
    if len(sxs_) > 8:
        # 先把四边形摆正，再量高度，才不会把倾斜算进字高
        qw, qh = quad_size(quad)
        st.ink_height = float(max(4.0, min(qh, (sys_.max() - sys_.min() + 1))))
        st.ink_width = float(max(4.0, (sxs_.max() - sxs_.min() + 1)))
    else:
        qw, qh = quad_size(quad)
        st.ink_height = float(max(4.0, qh * 0.78))
        st.ink_width = float(max(4.0, qw))

    # ④ 粗细：笔画像素 / 墨迹外接框面积
    if len(sxs_) > 8:
        ink_area = float(len(sxs_))
        box_area = float((sxs_.max() - sxs_.min() + 1) * (sys_.max() - sys_.min() + 1))
        occ = ink_area / (box_area + 1e-6)
        n_chars = max(1, len(sample_text)) if sample_text else 1
        # 单字占空比经验阈值：中文常规约 0.16~0.26，粗体 > 0.30
        st.bold = occ > (0.30 if has_cjk(sample_text) else 0.26)

    # ⑤ 倾斜角
    st.angle = quad_angle(quad)

    # ⑦ 锐度与噪点：取文字框外围一圈背景来测量
    pad = int(max(6, st.ink_height * 0.5))
    ex0, ey0 = max(0, x0 - pad), max(0, y0 - pad)
    ex1, ey1 = min(w, x1 + pad), min(h, y1 + pad)
    ring = cv2.cvtColor(image_rgb[ey0:ey1, ex0:ex1], cv2.COLOR_RGB2GRAY)
    st.blur_sigma = estimate_blur_sigma(ring)
    st.noise_sigma = estimate_noise_sigma(ring)

    # 字体：先按是否含中文挑一个能用的；若已知原文字，再在系统字体里
    # 选与原字笔画最像的，让替换后的字体观感与原图一致
    st.font_path = pick_font(sample_text or "中Aa", font_hint)
    if sample_text:
        matched = match_font_to_original(image_rgb, quad, sample_text,
                                         st.ink_height, st.angle)
        if matched:
            st.font_path = matched
    return st


# --------------------------------------------------------------------------- #
# 4. 新文字渲染与融合
# --------------------------------------------------------------------------- #

def _extend_quad(quad: np.ndarray, scale_w: float) -> np.ndarray:
    """沿文本方向按比例延展四边形（新文字比原文字长时使用）。

    保持左边缘与上下边缘的透视关系不变，只把右边缘沿"上边方向"外推，
    这样延展出来的部分依然落在原文字所在的那个平面上。
    """
    tl, tr, br, bl = quad
    top_v = (tr - tl) * scale_w
    bot_v = (br - bl) * scale_w
    return np.array([tl, tl + top_v, bl + bot_v, bl], dtype=np.float32)


def render_new_text(image_rgb: np.ndarray, new_text: str, style: TextStyle,
                    fit_mode: str = "fit_box",
                    color_override: Optional[Tuple[int, int, int]] = None,
                    font_override: Optional[str] = None,
                    size_scale: float = 1.0,
                    opacity: float = 1.0,
                    softness: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """渲染新文字并融合进图像。

    参数
    ----
    fit_mode : ``fit_box``  —— 缩放字号，让新文字恰好填满原四边形
               ``keep_size`` —— 保持原字号，四边形随文字长度自动延展
    返回
    ----
    (合成后的图像, 本次改动的掩膜)
    """
    if not new_text:
        return image_rgb.copy(), np.zeros(image_rgb.shape[:2], np.uint8)

    h, w = image_rgb.shape[:2]
    quad = order_quad(style.quad)
    qw, qh = quad_size(quad)
    color = color_override or style.fg_color
    font_path = font_override or style.font_path or pick_font(new_text)

    # ---- 决定字号 ----
    target_h = max(4.0, style.ink_height * size_scale)
    if fit_mode == "fit_box":
        font, size = fit_font_size(new_text, font_path,
                                   target_w=qw * 0.995, target_h=target_h)
    else:
        # keep_size：只按高度定字号，不限制宽度
        font, size = fit_font_size(new_text, font_path,
                                   target_w=1e6, target_h=target_h)
    size = max(4, int(round(size)))

    # ---- 渲染（4× 超采样） ----
    stroke_w = 1 if style.bold and size >= 18 else 0
    rgba = render_text_rgba(new_text, font_path, size, tuple(int(c) for c in color),
                            supersample=4, stroke_w=stroke_w,
                            letter_spacing=style.letter_spacing)
    if rgba.size == 0:
        return image_rgb.copy(), np.zeros((h, w), np.uint8)

    rh, rw = rgba.shape[:2]

    # ---- 决定目标四边形 ----
    if fit_mode == "keep_size":
        # 让四边形宽度正比于渲染宽度（相对原墨迹宽度）
        ratio = (rw / max(1.0, style.ink_width))
        dst_quad = _extend_quad(quad, max(0.15, ratio * (style.ink_width / max(1.0, qw))))
    else:
        dst_quad = quad.copy()

    # 在目标四边形内做垂直居中：把渲染图按比例贴到一张与四边形同宽高比的画布上
    dqw, dqh = quad_size(dst_quad)
    canvas_w = max(1, int(round(dqw)))
    canvas_h = max(1, int(round(dqh)))
    canvas = np.zeros((canvas_h, canvas_w, 4), np.uint8)

    fit = min(canvas_w / rw, canvas_h / rh, 1.0) if fit_mode == "fit_box" else \
        min(canvas_h / rh, 1.0) if rh > canvas_h else 1.0
    nw, nh = max(1, int(rw * fit)), max(1, int(rh * fit))
    resized = cv2.resize(rgba, (nw, nh), interpolation=cv2.INTER_LANCZOS4)

    ox = max(0, (canvas_w - nw) // 2) if fit_mode == "fit_box" else 0
    oy = max(0, (canvas_h - nh) // 2)
    ex, ey = min(canvas_w, ox + nw), min(canvas_h, oy + nh)
    canvas[oy:ey, ox:ex] = resized[:ey - oy, :ex - ox]

    # ---- ⑥ 透视变形到目标四边形 ----
    warped = warp_rgba_to_quad(canvas, dst_quad, (h, w))

    # ---- ⑦ 质感匹配：锐度 + 噪点 ----
    # 文字默认保持锐利——原图里的文字通常比照片背景更清晰，
    # 盲目把新字按"背景锐度"做高斯模糊（旧逻辑）会把字糊成一团。
    # 只有当用户显式要求"柔化"时才按背景清晰度做轻微模糊，并限制上限。
    blur_target = min(style.blur_sigma, 0.6) * max(0.0, softness)
    if blur_target > 0.05:
        warped = cv2.GaussianBlur(warped, (0, 0), blur_target)
    if style.noise_sigma > 0.3:
        rgb_part = add_grain(warped[:, :, :3], style.noise_sigma, seed=7)
        warped = np.dstack([rgb_part, warped[:, :, 3]])

    # ---- 不透明度 ----
    op = float(np.clip(opacity * style.opacity, 0.0, 1.0))
    if op < 0.999:
        warped[:, :, 3] = (warped[:, :, 3].astype(np.float32) * op).astype(np.uint8)

    out = alpha_composite(image_rgb, warped)
    changed = (warped[:, :, 3] > 4).astype(np.uint8) * 255
    return out, changed


# --------------------------------------------------------------------------- #
# 5. 一站式：替换某段文字
# --------------------------------------------------------------------------- #

def replace_text(image_rgb: np.ndarray, box: TextBox, new_text: str,
                 service: InpaintService,
                 fit_mode: str = "fit_box",
                 tight_erase: bool = True,
                 color_override: Optional[Tuple[int, int, int]] = None,
                 font_override: Optional[str] = None,
                 size_scale: float = 1.0,
                 style: Optional[TextStyle] = None,
                 softness: float = 0.0
                 ) -> Tuple[np.ndarray, TextStyle]:
    """把 ``box`` 位置的原文字替换为 ``new_text``。

    步骤：估计风格 → 擦除原文字 → 渲染新文字并融合。
    """
    st = style or estimate_style(image_rgb, box.quad, font_override, box.text)
    LOG.info("文字替换：'%s' → '%s' ｜ %s", box.text, new_text, st.describe())

    erased = erase_text(image_rgb, [box.quad], service, tight=tight_erase)
    out, _ = render_new_text(erased, new_text, st, fit_mode=fit_mode,
                             color_override=color_override,
                             font_override=font_override,
                             size_scale=size_scale,
                             softness=softness)
    return out, st


def batch_replace(image_rgb: np.ndarray,
                  items: Sequence[Tuple[TextBox, str]],
                  service: InpaintService, **kw) -> np.ndarray:
    """批量替换多段文字（例如整页 PDF 的术语替换）。

    先一次性擦掉所有原文字（掩膜合并，只跑一趟修复，省时间），
    再逐段渲染新文字。
    """
    if not items:
        return image_rgb.copy()

    quads = [b.quad for b, _ in items]
    styles = [estimate_style(image_rgb, b.quad, kw.get("font_override"), b.text)
              for b, _ in items]
    out = erase_text(image_rgb, quads, service,
                     tight=kw.get("tight_erase", True))
    for (box, new_text), st in zip(items, styles):
        if not new_text:
            continue
        out, _ = render_new_text(out, new_text, st,
                                 fit_mode=kw.get("fit_mode", "fit_box"),
                                 color_override=kw.get("color_override"),
                                 font_override=kw.get("font_override"),
                                 size_scale=kw.get("size_scale", 1.0),
                                 softness=kw.get("softness", 0.0))
    return out
