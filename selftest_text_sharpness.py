# -*- coding: utf-8 -*-
"""
selftest_text_sharpness.py —— 文字替换锐度回归测试（不依赖模型 / 界面 / OCR）
================================================================================

为什么需要它
------------
历史修复：文字替换默认会把"新字"按"背景锐度"做无条件高斯模糊，导致新字被糊成一团
（用户反馈"文字替换后字体太模糊了"）。修复后，模糊被改为默认关闭、仅当用户显式
拉高"柔化"滑块时才按上限轻微施加。

本测试守护这条不变量，防止将来有人把无条件模糊重新改回来：

  1. softness=0 时，无论背景锐度(blur_sigma)多大，新字都应保持锐利
     —— 即 sharpness(soft=0, 高blur) ≈ sharpness(soft=0, blur=0)
  2. softness>0 时，新字才被轻微模糊，且 sharpness(soft=0) 必须明显大于
     sharpness(soft=1)（默认 vs 柔化对比）

只用到 core.text_edit.render_new_text + utils，不加载 LaMa/SD/OCR，秒级跑完，
适合进 CI 与本地快速自测。

运行::

    python selftest_text_sharpness.py
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cv2
import numpy as np

from core.text_edit import TextStyle, render_new_text
from utils import LOG, pick_font


# --------------------------------------------------------------------------- #
def make_background(w: int = 960, h: int = 320) -> np.ndarray:
    """合成一张带渐变 + 轻噪点的平滑背景（文字要替换到这张图上）。"""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = 90 + 110 * (yy / h)
    g = 130 + 70 * (xx / w)
    b = 190 - 60 * (yy / h)
    img = np.dstack([r, g, b]).astype(np.uint8)
    rng = np.random.default_rng(7)
    img = np.clip(img + rng.normal(0, 1.5, img.shape), 0, 255).astype(np.uint8)
    return img


def sharpness(img: np.ndarray, mask: np.ndarray) -> float:
    """用拉普拉斯方差衡量掩膜区域内边缘的锐利程度（越大越清晰）。"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    m = mask > 0
    if int(m.sum()) < 50:
        return 0.0
    return float(lap[m].var())


def make_style(quad: np.ndarray, blur_sigma: float) -> TextStyle:
    """构造一个可控的 TextStyle（不调用 estimate_style，避免依赖修复引擎）。"""
    st = TextStyle(quad=quad.astype(np.float32))
    st.ink_height = 70.0
    st.ink_width = 560.0
    st.fg_color = (20, 20, 30)
    st.font_path = pick_font("全新标题 NEW 2027")
    st.blur_sigma = blur_sigma      # 控制"背景锐度"估计值
    st.noise_sigma = 0.0
    st.bold = False
    st.letter_spacing = 0.0
    st.opacity = 1.0
    return st


def main() -> int:
    print("=" * 70)
    print("智能图像编辑器 · 文字替换锐度回归测试")
    print("=" * 70)

    bg = make_background()
    quad = np.array([[80, 90], [640, 90], [640, 220], [80, 220]], np.float32)
    new_text = "全新标题 NEW 2027"

    # --- 不变量 1：softness=0 应无视背景锐度，保持锐利 ------------------- #
    st_sharp = make_style(quad, blur_sigma=2.0)   # 背景"很糊"的极端估计
    out_default, changed_default = render_new_text(
        bg, new_text, st_sharp, fit_mode="fit_box", softness=0.0)
    out_ideal, _ = render_new_text(
        bg, new_text, make_style(quad, blur_sigma=0.0),
        fit_mode="fit_box", softness=0.0)

    s_default = sharpness(out_default, changed_default)
    s_ideal = sharpness(out_ideal, (out_ideal != bg).any(axis=2).astype(np.uint8) * 255)

    print(f"  [softness=0, 背景估计blur=2.0] 锐度 = {s_default:.1f}")
    print(f"  [softness=0, 背景blur=0.0 基准] 锐度 = {s_ideal:.1f}")
    ratio_a = s_default / s_ideal if s_ideal > 0 else 0.0
    ok_a = 0.8 <= ratio_a <= 1.25
    print(f"  [{'PASS' if ok_a else 'FAIL'}] softness=0 应≈理想锐利 "
          f"(比值 {ratio_a:.2f}，须在 0.80~1.25)")

    # --- 不变量 2：softness>0 才轻微柔化，且默认明显更锐 ---------------- #
    out_soft, changed_soft = render_new_text(
        bg, new_text, make_style(quad, blur_sigma=2.0),
        fit_mode="fit_box", softness=1.0)
    s_soft = sharpness(out_soft, changed_soft)
    print(f"  [softness=1.0, 柔化上限]        锐度 = {s_soft:.1f}")

    ok_b = s_default > 1.5 * s_soft and s_soft > 0
    print(f"  [{'PASS' if ok_b else 'FAIL'}] 默认(soft=0)应明显比柔化(soft=1)锐利 "
          f"({s_default:.1f} > 1.5×{s_soft:.1f})")

    # --- 一致性：默认路径必须真的产生了文字（覆盖像素非空） ------------- #
    ok_c = int((changed_default > 0).sum()) > 500
    print(f"  [{'PASS' if ok_c else 'FAIL'}] 默认路径确实渲染出新文字 "
          f"({int((changed_default>0).sum())} px)")

    all_ok = ok_a and ok_b and ok_c
    print("\n" + "=" * 70)
    print("文字锐度回归测试：" + ("全部通过 ✓" if all_ok else "存在失败 ✗"))
    print("=" * 70)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
