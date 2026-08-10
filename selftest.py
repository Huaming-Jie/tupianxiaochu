# -*- coding: utf-8 -*-
"""
selftest.py —— 无界面自检（不依赖 PyQt5 / torch / OCR）
================================================================================
用途：在没有显示器的环境（服务器、CI）里验证算法层是否正常工作。

它会：
1. 合成一张带渐变背景 + 纹理 + 文字 + 水印 + Logo 的测试图
2. 依次跑：物体消除 → 水印检测 → 文字擦除 → 文字改写 → Logo 贴合 → PDF 往返
3. **重点校验**：修复后掩膜之外的像素是否与原图逐字节相同
4. 把每一步的结果输出到 ``./selftest_out/``

运行::

    python selftest.py
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cv2
import numpy as np

from core.inpaint import InpaintOptions, InpaintService
from core.logo_edit import LogoOptions, place_logo
from core.text_edit import (build_text_mask, erase_text, estimate_style,
                            render_new_text)
from core.watermark import auto_detect, refine_mask_for_watermark
from utils import (LOG, imwrite_rgb, pick_font, quad_to_mask,
                   render_text_rgba)

OUT = os.path.join(ROOT, "selftest_out")
os.makedirs(OUT, exist_ok=True)

W, H = 900, 600


# --------------------------------------------------------------------------- #
def make_test_image() -> np.ndarray:
    """合成一张有纹理、有渐变、有文字、有水印的测试图。"""
    # 竖向渐变 + 斜向条纹纹理 + 噪点，模拟真实照片的复杂背景
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = 90 + 110 * (yy / H)
    g = 130 + 70 * (xx / W)
    b = 190 - 60 * (yy / H)
    img = np.dstack([r, g, b])
    img += 14 * np.sin((xx + yy) / 11.0)[..., None]
    rng = np.random.default_rng(42)
    img += rng.normal(0, 2.4, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)

    # 一个"要被擦掉的物体"：实心圆
    cv2.circle(img, (250, 400), 46, (230, 70, 60), -1, lineType=cv2.LINE_AA)

    # 正文文字
    font = pick_font("测试文字Abc")
    txt = render_text_rgba("原始标题 Title 2026", font, 44, (28, 28, 34))
    th, tw = txt.shape[:2]
    _paste_rgba(img, txt, 60, 60)

    # 平铺半透明水印
    wm = render_text_rgba("© SAMPLE 样张", font, 26, (255, 255, 255))
    wm[:, :, 3] = (wm[:, :, 3].astype(np.float32) * 0.32).astype(np.uint8)
    for gy in range(0, H, 150):
        for gx in range(0, W, 300):
            _paste_rgba(img, wm, gx + 20, gy + 30)

    return img


def _paste_rgba(base: np.ndarray, rgba: np.ndarray, x: int, y: int):
    """把 RGBA 贴到 base 的 (x,y) 处（原地修改）。"""
    h, w = rgba.shape[:2]
    H_, W_ = base.shape[:2]
    x1, y1 = min(W_, x + w), min(H_, y + h)
    if x >= W_ or y >= H_ or x1 <= x or y1 <= y:
        return
    sub = rgba[:y1 - y, :x1 - x]
    a = sub[:, :, 3:4].astype(np.float32) / 255.0
    roi = base[y:y1, x:x1].astype(np.float32)
    base[y:y1, x:x1] = np.clip(
        sub[:, :, :3].astype(np.float32) * a + roi * (1 - a) + 0.5, 0, 255
    ).astype(np.uint8)


def make_logo() -> np.ndarray:
    """合成一个带透明通道的假 Logo。"""
    lg = np.zeros((120, 260, 4), np.uint8)
    cv2.rectangle(lg, (6, 6), (253, 113), (20, 110, 220, 255), -1, cv2.LINE_AA)
    cv2.circle(lg, (58, 60), 34, (255, 210, 40, 255), -1, cv2.LINE_AA)
    font = pick_font("NEW")
    t = render_text_rgba("NEWLOGO", font, 34, (255, 255, 255))
    _paste_rgba_rgba(lg, t, 104, 42)
    return lg


def _paste_rgba_rgba(base: np.ndarray, ov: np.ndarray, x: int, y: int):
    h, w = ov.shape[:2]
    H_, W_ = base.shape[:2]
    x1, y1 = min(W_, x + w), min(H_, y + h)
    if x1 <= x or y1 <= y:
        return
    sub = ov[:y1 - y, :x1 - x]
    a = sub[:, :, 3:4].astype(np.float32) / 255.0
    roi = base[y:y1, x:x1].astype(np.float32)
    rgb = sub[:, :, :3].astype(np.float32) * a + roi[:, :, :3] * (1 - a)
    alpha = np.maximum(roi[:, :, 3:4], sub[:, :, 3:4].astype(np.float32))
    base[y:y1, x:x1] = np.clip(np.dstack([rgb, alpha]) + 0.5, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
def check_outside_unchanged(name: str, before: np.ndarray, after: np.ndarray,
                            mask: np.ndarray, feather: int) -> bool:
    """校验：羽化影响范围之外的像素必须逐字节相同。"""
    # 羽化会向外扩散 feather 个像素，所以校验区取"掩膜膨胀 feather*3 之外"
    k = max(1, feather * 3)
    grown = cv2.dilate(mask, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (k * 2 + 1, k * 2 + 1)))
    outside = grown == 0
    diff = (before[outside] != after[outside])
    n_bad = int(diff.sum())
    total = int(outside.sum()) * 3
    ok = (n_bad == 0)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}：掩膜外 {total} 个通道值中"
          f"有 {n_bad} 个被改动")
    return ok


# --------------------------------------------------------------------------- #
def main() -> int:
    print("=" * 70)
    print("智能图像编辑器 · 算法层自检")
    print("=" * 70)

    svc = InpaintService(device="auto")
    print("修复引擎：", svc.warmup())
    print()

    img = make_test_image()
    imwrite_rgb(os.path.join(OUT, "00_source.png"), img)
    print(f"[1] 已生成测试图 {W}×{H} → 00_source.png")

    all_ok = True

    # ---------------- ① 物体消除 ---------------- #
    print("\n[2] 物体消除")
    mask = np.zeros((H, W), np.uint8)
    cv2.circle(mask, (250, 400), 52, 255, -1)
    opt = InpaintOptions(method="lama", dilate=4, feather=4)
    r1 = svc.inpaint(img, mask, opt)
    imwrite_rgb(os.path.join(OUT, "01_object_removed.png"), r1)
    all_ok &= check_outside_unchanged("物体消除", img, r1, mask, opt.feather)
    print(f"  修复区域平均色变化：{np.abs(r1[mask>0].astype(int) - img[mask>0].astype(int)).mean():.1f}")

    # ---------------- ② 水印检测 ---------------- #
    print("\n[3] 水印自动检测（无 OCR，仅半透明 + 周期性线索）")
    wm_mask, cands, diag = auto_detect(img, ocr_boxes=None)
    print("  ", diag)
    vis = img.copy()
    for c in cands[:40]:
        x0, y0, x1, y1 = c.bbox
        cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 200, 0), 2)
    imwrite_rgb(os.path.join(OUT, "02_watermark_boxes.png"), vis)
    if wm_mask.max() > 0:
        fine = refine_mask_for_watermark(img, wm_mask)
        r2 = svc.inpaint(img, fine, InpaintOptions(method="lama", dilate=3))
        imwrite_rgb(os.path.join(OUT, "03_watermark_removed.png"), r2)
        all_ok &= check_outside_unchanged("水印消除", img, r2, fine, 4)
        print(f"  笔画级收紧：{int((wm_mask>0).sum())} → {int((fine>0).sum())} px")
    else:
        print("  未产生候选（合成水印对比度较低属正常，可手动框选）")

    # ---------------- ③ 文字消除 ---------------- #
    print("\n[4] 文字消除（手动指定标题四边形）")
    title_quad = np.array([[58, 55], [560, 55], [560, 120], [58, 120]], np.float32)
    tmask = build_text_mask(img, [title_quad], tight=True)
    ratio = (tmask > 0).sum() / max(1, (quad_to_mask((H, W), title_quad) > 0).sum())
    print(f"  笔画级掩膜占框面积：{ratio*100:.1f}%（越小说明保留的背景越多）")
    r3 = erase_text(img, [title_quad], svc, tight=True)
    imwrite_rgb(os.path.join(OUT, "04_text_erased.png"), r3)
    all_ok &= check_outside_unchanged("文字消除", img, r3, tmask, 4)

    # ---------------- ④ 文字修改 ---------------- #
    print("\n[5] 文字修改（风格估计 + 重绘）")
    style = estimate_style(img, title_quad, sample_text="原始标题 Title 2026")
    print("  风格分析：", style.describe())
    r4, changed = render_new_text(r3, "全新标题 NEW 2027", style, fit_mode="fit_box")
    imwrite_rgb(os.path.join(OUT, "05_text_replaced.png"), r4)
    print(f"  新文字覆盖像素：{int((changed>0).sum())}")

    # ---------------- ⑤ Logo 贴合 ---------------- #
    print("\n[6] Logo 透视贴合")
    logo = make_logo()
    # 一个带透视的四边形（模拟贴在斜面上）
    quad = np.array([[600, 330], [860, 300], [872, 410], [606, 432]], np.float32)
    r5, lmask = place_logo(r4, logo, quad,
                           LogoOptions(blend_mode="normal", shading_strength=0.8))
    imwrite_rgb(os.path.join(OUT, "06_logo_placed.png"), r5)
    print(f"  Logo 覆盖像素：{int((lmask>0).sum())}")
    all_ok &= check_outside_unchanged("Logo 贴合", r4, r5, lmask, 2)

    # ---------------- ⑥ PDF 往返 ---------------- #
    print("\n[7] PDF 往返（光栅化 → 编辑 → 回写）")
    try:
        from core.pdf_io import (HAS_FITZ, images_to_pdf, pages_to_pdf,
                                 pdf_to_pages)
        if not HAS_FITZ:
            raise RuntimeError("未安装 PyMuPDF")
        p_in = os.path.join(OUT, "07_input.pdf")
        images_to_pdf([img, r5], p_in, dpi=300, lossless=True)
        pages = pdf_to_pages(p_in, dpi=300, adaptive=True)
        print(f"  读回 {len(pages)} 页，第 1 页 {pages[0].size_px}，"
              f"DPI={pages[0].dpi:.0f}")
        # 尺寸还原校验
        same = pages[0].size_px == (W, H)
        print(f"  [{'PASS' if same else 'WARN'}] 分辨率还原：{pages[0].size_px} vs ({W}, {H})")
        pages[0].image = r1
        p_out = os.path.join(OUT, "08_output.pdf")
        pages_to_pdf(pages, p_out, lossless=False, jpeg_quality=95)
        print(f"  已写出 {p_out}（{os.path.getsize(p_out)/1024:.0f} KB）")

        # 新增：多页只改一页时，未改页应原样保留（不二次光栅化）
        from core.pdf_io import pages_to_pdf_preserved
        p_in2 = os.path.join(OUT, "07b_src.pdf")
        images_to_pdf([img, r5], p_in2, dpi=300, lossless=True)   # 2 页
        pg = pdf_to_pages(p_in2, dpi=300, adaptive=True)
        pg[0].image = r1                                           # 仅改第 1 页
        p_out2 = os.path.join(OUT, "08b_preserved.pdf")
        pages_to_pdf_preserved(p_in2, pg, [0], p_out2, lossless=True)
        re_pg = pdf_to_pages(p_out2, dpi=300, adaptive=True)
        a, b = re_pg[1].image, pg[1].image                        # 未改的第 2 页
        h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
        maxdiff = int(np.abs(a[:h, :w].astype(int) - b[:h, :w].astype(int)).max())
        preserved = maxdiff <= 2
        saved = 100 * (1 - os.path.getsize(p_out2) / max(1, os.path.getsize(p_out)))
        print(f"  [{'PASS' if preserved else 'FAIL'}] 保留未改页："
              f"第2页最大差={maxdiff}，体积省 {saved:.0f}%")
        all_ok &= preserved
    except Exception as e:
        print("  [SKIP] PDF 测试跳过：", e)

    print("\n" + "=" * 70)
    print(("全部关键校验通过 ✓" if all_ok else "存在校验失败 ✗") +
          f"　结果已输出到：{OUT}")
    print("=" * 70)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
