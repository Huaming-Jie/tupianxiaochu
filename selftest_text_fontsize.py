# -*- coding: utf-8 -*-
"""
selftest_text_fontsize.py —— 文字替换"字体一致 + 大小一致"回归测试
========================================================================

守护两条需求（用户提出）：
  1. 替换后的字体应与原字体一致 → 在系统可用字体里挑与原字笔画最像的
     （match_font_to_original，按 IoU 比对），而不是永远掉到通用默认字体。
  2. 替换后的大小应与原字一致 → 新字高度应≈原字墨迹高度（fit_box 锁定）。

只用到 core.text_edit + utils，不加载 LaMa/SD/OCR，秒级跑完，
适合进 CI 与本地快速自测。

运行::

    python selftest_text_fontsize.py
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cv2
import numpy as np

from core.text_edit import (estimate_style, render_new_text,
                            stroke_mask_in_quad)
from utils import list_available_fonts, pick_font, render_text_rgba


# --------------------------------------------------------------------------- #
def make_background(w: int = 960, h: int = 320) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = 90 + 110 * (yy / h)
    g = 130 + 70 * (xx / w)
    b = 190 - 60 * (yy / h)
    img = np.dstack([r, g, b]).astype(np.uint8)
    rng = np.random.default_rng(11)
    img = np.clip(img + rng.normal(0, 1.5, img.shape), 0, 255).astype(np.uint8)
    return img


def _paste(base: np.ndarray, rgba: np.ndarray, x: int, y: int):
    h, w = rgba.shape[:2]
    H, W = base.shape[:2]
    x1, y1 = min(W, x + w), min(H, y + h)
    if x >= W or y >= H or x1 <= x or y1 <= y:
        return
    sub = rgba[:y1 - y, :x1 - x]
    a = sub[:, :, 3:4].astype(np.float32) / 255.0
    roi = base[y:y1, x:x1].astype(np.float32)
    base[y:y1, x:x1] = np.clip(
        sub[:, :, :3].astype(np.float32) * a + roi * (1 - a) + 0.5, 0, 255
    ).astype(np.uint8)


def ink_height(mask: np.ndarray) -> float:
    ys, _ = np.where(mask > 0)
    return float(ys.max() - ys.min() + 1) if len(ys) else 0.0


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a > 0, b > 0).sum())
    union = int(np.logical_or(a > 0, b > 0).sum())
    return float(inter) / float(union + 1e-6)


def _norm(mask: np.ndarray, n: int = 220) -> Optional[np.ndarray]:
    """裁到笔画外接框再拉伸到固定画布（与 match_font_to_original 同口径）。"""
    ys, xs = np.where(mask > 0)
    if len(xs) < 4:
        return None
    sub = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return cv2.resize(sub, (n, n), interpolation=cv2.INTER_NEAREST)


# --------------------------------------------------------------------------- #
def main() -> int:
    print("=" * 70)
    print("智能图像编辑器 · 文字字体/大小一致性回归测试")
    print("=" * 70)

    bg = make_background()
    quad = np.array([[80, 90], [700, 90], [700, 220], [80, 220]], np.float32)
    new_text = "原始标题 Title 2026"

    # 用"本机第一个可用中文字体"渲染原字，模拟真实原图
    default_font = pick_font(new_text)
    # 主动挑一个"非默认"的候选字体作为原图真字体，让比对必须真正生效：
    # 若匹配被关掉而退回默认字体，下面的字体一致性断言就会失败（形成回归守卫）
    src_font = None
    for c in list_available_fonts(cjk=True):
        if os.path.basename(c) != os.path.basename(default_font):
            src_font = c
            break
    if src_font is None:
        src_font = default_font
    font = src_font
    print(f"  用于合成原图的字体（非默认）：{os.path.basename(font)}")
    orig_rgba = render_text_rgba(new_text, font, 60, (24, 24, 32))
    _paste(bg, orig_rgba, 80, 90)

    orig_mask = stroke_mask_in_quad(bg, quad, dilate=0)
    h0 = ink_height(orig_mask)
    print(f"  原字墨迹高度 = {h0:.1f} px")

    # ---- 估计风格（带原文字，应触发字体比对）---- #
    st = estimate_style(bg, quad, sample_text=new_text)
    print(f"  估计风格：{st.describe()}")
    print(f"  选中字体：{os.path.basename(st.font_path)}")

    all_ok = True

    # --- 需求 2：大小一致 --- #
    # 估计出的墨迹高度应≈实际原字高度
    ok_h_est = abs(st.ink_height - h0) <= 0.15 * h0
    print(f"  [{'PASS' if ok_h_est else 'FAIL'}] 估计高度≈原字高度 "
          f"({st.ink_height:.1f} vs {h0:.1f})")

    # 用同样文字重绘后，新字高度应≈原字高度
    out, changed = render_new_text(bg, new_text, st, fit_mode="fit_box",
                                   softness=0.0)
    h1 = ink_height(changed)
    ok_h_new = abs(h1 - h0) <= 0.12 * h0
    print(f"  [{'PASS' if ok_h_new else 'FAIL'}] 重绘后高度≈原字高度 "
          f"({h1:.1f} vs {h0:.1f})")
    all_ok &= ok_h_est and ok_h_new

    # --- 需求 1：字体一致（选中字体应复现原字笔画）--- #
    # 关键：用"合成原图时真正使用的字体(font)"作为黄金标准，比较各字体复现原字的能力。
    # （在渐变背景上做前景提取会引入噪声，绝对 IoU 基线本身≈0.58，故用相对判据。）
    src_rgba = render_text_rgba(new_text, font, int(round(st.ink_height)), (0, 0, 0))
    src_mask = (src_rgba[:, :, 3] > 0).astype(np.uint8)
    iou_source = mask_iou(_norm(orig_mask), _norm(src_mask))   # 黄金标准得分

    matched_rgba = render_text_rgba(new_text, st.font_path,
                                    int(round(st.ink_height)), (0, 0, 0))
    matched_mask = (matched_rgba[:, :, 3] > 0).astype(np.uint8)
    iou_matched = mask_iou(_norm(orig_mask), _norm(matched_mask))

    default_font = pick_font(new_text)
    default_rgba = render_text_rgba(new_text, default_font,
                                    int(round(st.ink_height)), (0, 0, 0))
    default_mask = (default_rgba[:, :, 3] > 0).astype(np.uint8)
    iou_default = mask_iou(_norm(orig_mask), _norm(default_mask))

    # 选中字体复现原字的能力，应不弱于"原图真字体"本身（说明挑对或挑到等价字体）
    ok_font = iou_matched >= iou_source - 0.05
    ok_not_worse = iou_matched >= iou_default - 0.02
    same_as_source = (os.path.basename(st.font_path) == os.path.basename(font))
    print(f"  黄金标准(原字体)IoU={iou_source:.2f}｜选中字体IoU={iou_matched:.2f}｜"
          f"默认IoU={iou_default:.2f}")
    print(f"  [{'PASS' if ok_font else 'FAIL'}] 选中字体复现原字能力≈原字体本身 "
          f"(≥{iou_source - 0.05:.2f})")
    print(f"  [{'PASS' if ok_not_worse else 'FAIL'}] 字体比对不差于默认 "
          f"({iou_matched:.2f} vs {iou_default:.2f})")
    print(f"  [{'OK' if same_as_source else 'INFO'}] 选中字体与原图字体相同："
          f"{same_as_source}")
    all_ok &= ok_font and ok_not_worse

    # --- 守卫：不给原文字时仍应回退到一个存在的字体（不崩溃）--- #
    st2 = estimate_style(bg, quad, sample_text="")
    ok_fallback = os.path.isfile(st2.font_path)
    print(f"  [{'PASS' if ok_fallback else 'FAIL'}] 无原文字时回退到存在的字体 "
          f"({os.path.basename(st2.font_path)})")
    all_ok &= ok_fallback

    print("\n" + "=" * 70)
    print("文字字体/大小一致性回归测试：" + ("全部通过 ✓" if all_ok else "存在失败 ✗"))
    print("=" * 70)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
