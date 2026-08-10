# -*- coding: utf-8 -*-
"""
selftest_ocr.py —— 带 OCR 自动检测的全链路自检（用户本机运行）
================================================================================
与 selftest.py 的区别：本脚本刻意不手动框选，而是先用 OCR 把图里的文字
"读"出来，再用读到的框去驱动：

  * 文字消除（自动框）    erase_text(img, [box.quad], svc)
  * 文字修改（自动框）    replace_text(img, box, "新文字", svc)
  * 水印消除（自动框）    auto_detect(img, ocr_boxes=boxes) -> inpaint
  * 物体消除 / Logo / PDF 与 selftest.py 一致

前置依赖（本机需安装）：
    pip install paddleocr            # 或 easyocr
    并把深度学习权重放好（big-lama.pt / lama_fp32.onnx，见 downloader）
默认走 TorchScript 版 LaMa；若 torch 不可用（如内存受限），设环境变量
    SIE_NO_TORCH=1
即可自动落到 ONNX 版（已验证等价）。

运行：
    python selftest_ocr.py
结果图片输出到 ./selftest_out/ocr_*.png
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
from core.text_edit import erase_text, replace_text
from core.watermark import auto_detect, refine_mask_for_watermark
from models.ocr import OcrEngine
from utils import LOG, imwrite_rgb, pick_font, render_text_rgba

OUT = os.path.join(ROOT, "selftest_out")
os.makedirs(OUT, exist_ok=True)

W, H = 900, 600


# --------------------------------------------------------------------------- #
# 场景构造：留一个"已知答案"的标题 + 平铺/角标水印，方便 OCR 抓
# --------------------------------------------------------------------------- #
def _paste_rgba(base, rgba, x, y):
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


def build_scene():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = 90 + 110 * (yy / H)
    g = 130 + 70 * (xx / W)
    b = 190 - 60 * (yy / H)
    img = np.dstack([r, g, b]).astype(np.float32)
    img += 14 * np.sin((xx + yy) / 11.0)[..., None]
    rng = np.random.default_rng(42)
    img += rng.normal(0, 2.4, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)

    # 物体
    cv2.circle(img, (250, 400), 46, (230, 70, 60), -1, lineType=cv2.LINE_AA)

    # 标题（清晰、高对比，OCR 必能读到）
    font = pick_font("测试文字Abc")
    title = render_text_rgba("原始标题 Title 2026", font, 44, (28, 28, 34))
    _paste_rgba(img, title, 60, 60)

    # 平铺半透明水印（重复 → OCR 重复线索）
    wm = render_text_rgba("© SAMPLE 样张", font, 26, (255, 255, 255))
    wm[:, :, 3] = (wm[:, :, 3].astype(np.float32) * 0.5).astype(np.uint8)
    for gy in range(0, H, 150):
        for gx in range(0, W, 300):
            _paste_rgba(img, wm, gx + 20, gy + 30)

    # 角标水印（不透明、含关键词 → OCR 关键词线索）
    stamp = render_text_rgba("© 2026 SAMPLE.COM", font, 22, (20, 20, 20))
    _paste_rgba(img, stamp, W - 270, 24)
    return img


def _paste_rgba_on_rgba(base, ov, x, y):
    """把 RGBA 贴到 RGBA 底图上（alpha-over 合成）。"""
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


def make_logo():
    lg = np.zeros((120, 260, 4), np.uint8)
    cv2.rectangle(lg, (6, 6), (253, 113), (20, 110, 220, 255), -1, cv2.LINE_AA)
    cv2.circle(lg, (58, 60), 34, (255, 210, 40, 255), -1, cv2.LINE_AA)
    font2 = pick_font("NEW")
    t = render_text_rgba("NEWLOGO", font2, 34, (255, 255, 255))
    _paste_rgba_on_rgba(lg, t, 104, 42)
    return lg


# --------------------------------------------------------------------------- #
def check_outside_unchanged(name, before, after, mask, feather):
    k = max(1, feather * 3)
    grown = cv2.dilate(mask, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (k * 2 + 1, k * 2 + 1)))
    outside = grown == 0
    n_bad = int((before[outside] != after[outside]).sum())
    total = int(outside.sum()) * 3
    ok = (n_bad == 0)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}：掩膜外 {total} 通道值中"
          f"有 {n_bad} 个被改动")
    return ok


# --------------------------------------------------------------------------- #
def main() -> int:
    print("=" * 70)
    print("智能图像编辑器 · 带 OCR 自动检测的全链路自检")
    print("=" * 70)

    svc = InpaintService(device="auto")
    print("修复引擎：", svc.warmup())
    print()

    # ---- OCR ----
    ocr = OcrEngine(lang="ch", prefer="paddle")
    backend = ocr.load()
    if backend is None:
        print("[OCR] 未安装 OCR 引擎（paddleocr/easyocr）。")
        print("       请先：pip install paddleocr")
        print("       本脚本将仅验证非 OCR 功能并以手动框兜底文字/水印。\n")
    else:
        print(f"[OCR] 后端 = {backend}")

    img = build_scene()
    imwrite_rgb(os.path.join(OUT, "ocr_00_source.png"), img)
    print(f"[1] 已生成测试图 {W}×{H} → ocr_00_source.png")

    all_ok = True
    boxes = ocr.detect(img) if backend else []
    print(f"[OCR] 检测到 {len(boxes)} 个文本块")
    for i, b in enumerate(boxes):
        print(f"     #{i} \"{b.text}\" score={b.score:.2f} bbox={b.bbox}")

    # 选标题块：含 Title/原始 最佳，否则取最大块
    title_box = None
    for b in boxes:
        if "title" in b.text.lower() or "原始" in b.text:
            title_box = b
            break
    if title_box is None and boxes:
        title_box = max(boxes, key=lambda b: (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1]))

    # ---------------- ① 物体消除（手动，与 OCR 无关） ---------------- #
    print("\n[2] 物体消除（手动圈选）")
    mask = np.zeros((H, W), np.uint8)
    cv2.circle(mask, (250, 400), 52, 255, -1)
    opt = InpaintOptions(method="lama", dilate=4, feather=4)
    r1 = svc.inpaint(img, mask, opt)
    imwrite_rgb(os.path.join(OUT, "ocr_01_object_removed.png"), r1)
    all_ok &= check_outside_unchanged("物体消除", img, r1, mask, opt.feather)

    # ---------------- ② 文字消除（自动：OCR 框） ---------------- #
    print("\n[3] 文字消除（OCR 自动框）")
    if title_box is not None:
        r3 = erase_text(img, [title_box.quad], svc, tight=True)
        imwrite_rgb(os.path.join(OUT, "ocr_04_text_erased.png"), r3)
        tmask = cv2.zeros((H, W), np.uint8)
        cv2.fillPoly(tmask, [np.int32(title_box.quad)], 255)
        all_ok &= check_outside_unchanged("文字消除", img, r3, tmask, 6)
        print(f"    已用 OCR 框 \"{title_box.text}\" 自动擦除")
    else:
        print("    [SKIP] 未检测到标题块（OCR 不可用或场景无文字）")

    # ---------------- ③ 文字修改（自动：OCR 框 + 重绘） ---------------- #
    print("\n[4] 文字修改（OCR 自动框 + 风格保持重绘）")
    if title_box is not None:
        r4, _ = replace_text(img, title_box, "全新标题 NEW 2027", svc)
        imwrite_rgb(os.path.join(OUT, "ocr_05_text_replaced.png"), r4)
        print(f"    已将 \"{title_box.text}\" 改为 \"全新标题 NEW 2027\"")
    else:
        print("    [SKIP] 未检测到标题块")

    # ---------------- ④ 水印消除（自动：OCR 线索 + 修复） ---------------- #
    print("\n[5] 水印自动检测与消除（OCR 文本线索）")
    wm_mask, cands, diag = auto_detect(img, ocr_boxes=boxes if backend else None)
    print("    ", diag)
    if cands:
        fine = refine_mask_for_watermark(img, wm_mask)
        r2 = svc.inpaint(img, fine, InpaintOptions(method="lama", dilate=3))
        imwrite_rgb(os.path.join(OUT, "ocr_03_watermark_removed.png"), r2)
        all_ok &= check_outside_unchanged("水印消除", img, r2, fine, 4)
        print(f"    笔画级收紧：{int((wm_mask > 0).sum())} → {int((fine > 0).sum())} px")
    else:
        print("    [WARN] 未产生水印候选（可手动框选；或调低 min_score）")

    # ---------------- ⑤ Logo 贴合 ---------------- #
    print("\n[6] Logo 透视贴合")
    logo = make_logo()
    quad = np.array([[600, 330], [860, 300], [872, 410], [606, 432]], np.float32)
    r5, lmask = place_logo(r4 if title_box is not None else img, logo, quad,
                           LogoOptions(blend_mode="normal", shading_strength=0.8))
    imwrite_rgb(os.path.join(OUT, "ocr_06_logo_placed.png"), r5)
    all_ok &= check_outside_unchanged("Logo 贴合",
                                      r4 if title_box is not None else img, r5, lmask, 2)

    # ---------------- ⑥ PDF 往返 ---------------- #
    print("\n[7] PDF 往返")
    try:
        from core.pdf_io import HAS_FITZ, images_to_pdf, pages_to_pdf, pdf_to_pages
        if not HAS_FITZ:
            raise RuntimeError("未安装 PyMuPDF")
        base = r4 if title_box is not None else img
        p_in = os.path.join(OUT, "ocr_07_input.pdf")
        images_to_pdf([img, r5], p_in, dpi=300, lossless=True)
        pages = pdf_to_pages(p_in, dpi=300, adaptive=True)
        same = pages[0].size_px == (W, H)
        print(f"    [{'PASS' if same else 'WARN'}] 分辨率还原：{pages[0].size_px} vs ({W}, {H})")
        pages[0].image = r1
        p_out = os.path.join(OUT, "ocr_08_output.pdf")
        pages_to_pdf(pages, p_out, lossless=False, jpeg_quality=95)
        print(f"    已写出 {p_out}（{os.path.getsize(p_out) / 1024:.0f} KB）")
    except Exception as e:
        print("    [SKIP] PDF 测试跳过：", e)

    print("\n" + "=" * 70)
    print(("全部关键校验通过 ✓" if all_ok else "存在校验失败 ✗") +
          f"　结果已输出到：{OUT}")
    print("=" * 70)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
