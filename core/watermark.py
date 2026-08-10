# -*- coding: utf-8 -*-
"""
core/watermark.py —— 水印自动检测
================================================================================
先说清楚一件事：**通用水印检测没有银弹**。

学术上"水印去除"通常假设你有一批含同一水印的图片（可以做多图对齐求解
水印的 alpha 与 RGB），或者你已知水印模板。单张任意图片上的盲检测，
本质是一个欠定问题。所以这里的策略是 **多线索投票 + 人工确认**：

四条互补线索
------------
1. **重复文本线索**：OCR 结果里同一串文字出现 ≥2 次，且分布在图像不同位置
   → 极可能是平铺水印（小红书、图库站的典型做法）。
2. **版权符号线索**：文本含 © ® ™、``www.``、``.com``、"版权/水印/摄影"
   等关键词。
3. **半透明叠加线索**：水印笔画是"低对比度但边缘锐利"的细结构。
   用形态学 tophat/blackhat 提取细笔画，再用**局部对比度带通**筛掉
   真实物体的强边缘（真实边缘对比度高）与噪声（对比度极低）。
4. **周期性线索**：对高通残差做自相关（FFT 加速），若存在显著的非零位移
   峰值，说明画面上有周期性平铺图案 → 平铺水印。

最终把命中的线索取并集，输出 **候选框列表 + 建议掩膜**，
UI 上以红框呈现，用户勾选后再执行擦除。**永远不自动直接擦**——
误擦一张照片里的真实招牌，代价远大于让用户多点一下。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from utils import LOG, TextBox, order_quad, quad_to_mask

#: 版权/水印关键词与符号
_WM_KEYWORDS = re.compile(
    r"(©|®|™|copyright|版权|水印|摄影|图虫|视觉中国|全景|站酷|shutterstock|"
    r"getty|istock|www\.|\.com|\.cn|\.net|@|所有|禁止转载|preview|sample|样张)",
    re.IGNORECASE)


@dataclass
class WatermarkCandidate:
    """一个水印候选区域。"""
    quad: np.ndarray            # (4,2)
    score: float                # 0~1 置信度
    reason: str                 # 命中的线索说明
    text: str = ""

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        x0 = int(self.quad[:, 0].min()); y0 = int(self.quad[:, 1].min())
        x1 = int(self.quad[:, 0].max()); y1 = int(self.quad[:, 1].max())
        return x0, y0, x1, y1


# --------------------------------------------------------------------------- #
# 线索 1 & 2：基于 OCR 的文本线索
# --------------------------------------------------------------------------- #

def detect_from_text(boxes: Sequence[TextBox],
                     img_shape: Tuple[int, int]) -> List[WatermarkCandidate]:
    """从 OCR 结果中挑出疑似水印的文本。"""
    out: List[WatermarkCandidate] = []
    if not boxes:
        return out

    h, w = img_shape[:2]

    # --- 统计重复文本 ---
    norm = {}
    for b in boxes:
        key = re.sub(r"\s+", "", b.text).lower()
        if len(key) < 2:
            continue
        norm.setdefault(key, []).append(b)

    for key, group in norm.items():
        score, reasons = 0.0, []

        if len(group) >= 2:
            score += min(0.45, 0.18 * len(group))
            reasons.append(f"重复出现{len(group)}次")

        if _WM_KEYWORDS.search(key):
            score += 0.45
            reasons.append("含版权/域名关键词")

        for b in group:
            s, r = score, list(reasons)

            # 位置线索：贴边或角落（常见水印摆放）
            cx, cy = b.center
            near_edge = (cx < w * 0.18 or cx > w * 0.82 or
                         cy < h * 0.12 or cy > h * 0.88)
            if near_edge:
                s += 0.12
                r.append("位于边角")

            # 尺寸线索：水印一般不会占满画面
            x0, y0, x1, y1 = b.bbox
            area_ratio = ((x1 - x0) * (y1 - y0)) / float(w * h + 1e-6)
            if area_ratio > 0.35:
                s -= 0.3

            if s >= 0.4:
                out.append(WatermarkCandidate(b.quad.copy(), min(1.0, s),
                                              "、".join(r), b.text))
    return out


# --------------------------------------------------------------------------- #
# 线索 3：半透明细笔画检测
# --------------------------------------------------------------------------- #

def detect_translucent(image_rgb: np.ndarray,
                       stroke_px: int = 3,
                       lo: float = 4.0, hi: float = 42.0,
                       min_area_ratio: float = 2e-5,
                       max_area_ratio: float = 0.12) -> Tuple[np.ndarray, List[WatermarkCandidate]]:
    """检测半透明叠加的细笔画（文字型/线条型水印）。

    核心是一个 **带通** 判据：
    ``lo < 局部残差幅值 < hi``。
    真实物体边缘的残差远大于 hi，传感器噪声远小于 lo，
    夹在中间的多半就是"淡淡的一层"叠加物。
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (stroke_px * 2 + 1, stroke_px * 2 + 1))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)     # 比周围亮的细结构
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)  # 比周围暗的细结构
    resp = cv2.max(tophat, blackhat).astype(np.float32)

    band = ((resp > lo) & (resp < hi)).astype(np.uint8) * 255

    # 形态学聚合：把离散笔画连成"字块"
    band = cv2.morphologyEx(band, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5)),
                            iterations=2)
    band = cv2.morphologyEx(band, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    h, w = gray.shape
    total = float(h * w)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(band, 8)

    mask = np.zeros_like(band)
    cands: List[WatermarkCandidate] = []
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        ar = area / total
        if ar < min_area_ratio or ar > max_area_ratio:
            continue
        fill = area / float(cw * ch + 1e-6)
        if fill < 0.06 or fill > 0.92:      # 太稀疏或太实心都不像文字水印
            continue
        aspect = cw / float(ch + 1e-6)
        if aspect < 0.15 or aspect > 30:
            continue
        mask[labels == i] = 255
        quad = np.array([[x, y], [x + cw, y], [x + cw, y + ch], [x, y + ch]],
                        dtype=np.float32)
        cands.append(WatermarkCandidate(quad, 0.45, "半透明细笔画"))

    return mask, cands


# --------------------------------------------------------------------------- #
# 线索 4：周期性平铺检测
# --------------------------------------------------------------------------- #

def detect_periodicity(image_rgb: np.ndarray,
                       min_shift: int = 24) -> Tuple[bool, Tuple[int, int], float]:
    """用自相关判断画面是否存在周期性平铺图案。

    返回 (是否周期, (周期dx, 周期dy), 峰值强度)。

    实现：对高通残差做 FFT，再用 ``IFFT(|F|²)`` 得到自相关图
    （Wiener–Khinchin 定理），在排除原点邻域后找最大峰。
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # 缩放以加速（周期同比缩放，最后还原）
    scale = 1.0
    if max(gray.shape) > 1024:
        scale = 1024.0 / max(gray.shape)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    hp = gray - cv2.GaussianBlur(gray, (0, 0), 3.0)
    hp -= hp.mean()
    win = np.outer(np.hanning(hp.shape[0]), np.hanning(hp.shape[1]))
    hp = hp * win

    F = np.fft.rfft2(hp)
    ac = np.fft.irfft2(F * np.conj(F), s=hp.shape)
    ac = np.fft.fftshift(ac)
    ac /= (ac.max() + 1e-9)

    cy, cx = np.array(ac.shape) // 2
    r = max(min_shift, int(min(ac.shape) * 0.03))
    y, x = np.ogrid[:ac.shape[0], :ac.shape[1]]
    ac_masked = ac.copy()
    ac_masked[(x - cx) ** 2 + (y - cy) ** 2 < r * r] = 0

    py, px = np.unravel_index(int(np.argmax(ac_masked)), ac.shape)
    peak = float(ac_masked[py, px])
    dx, dy = int(abs(px - cx) / scale), int(abs(py - cy) / scale)
    return peak > 0.22, (dx, dy), peak


# --------------------------------------------------------------------------- #
# 统一入口
# --------------------------------------------------------------------------- #

def auto_detect(image_rgb: np.ndarray,
                ocr_boxes: Optional[Sequence[TextBox]] = None,
                use_translucent: bool = True,
                min_score: float = 0.4
                ) -> Tuple[np.ndarray, List[WatermarkCandidate], str]:
    """水印自动检测总入口。

    返回
    ----
    (建议掩膜 0/255, 候选列表, 诊断文本)
    """
    h, w = image_rgb.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cands: List[WatermarkCandidate] = []
    notes: List[str] = []

    # --- 文本线索 ---
    if ocr_boxes:
        tc = detect_from_text(ocr_boxes, (h, w))
        cands.extend(tc)
        notes.append(f"文本线索命中 {len(tc)} 处")

    # --- 周期性线索（只作为置信度加权，不单独产生框） ---
    periodic, period, peak = detect_periodicity(image_rgb)
    if periodic:
        notes.append(f"检测到周期平铺（周期≈{period[0]}×{period[1]}px，强度{peak:.2f}）")
        for c in cands:
            c.score = min(1.0, c.score + 0.15)
            c.reason += "、周期平铺"

    # --- 半透明线索 ---
    if use_translucent:
        tm, tcands = detect_translucent(image_rgb)
        if periodic:                      # 有周期性时，半透明线索更可信
            for c in tcands:
                c.score += 0.2
        # 半透明候选若与文本候选重叠，合并置信度而非重复列出
        for c in tcands:
            if not _overlaps_any(c, cands, 0.4):
                cands.append(c)
        notes.append(f"半透明线索命中 {len(tcands)} 处")

    # --- 汇总掩膜 ---
    kept = [c for c in cands if c.score >= min_score]
    for c in kept:
        mask = cv2.bitwise_or(mask, quad_to_mask((h, w), c.quad))

    kept.sort(key=lambda c: -c.score)
    diag = "；".join(notes) + f"；最终保留 {len(kept)} 个候选"
    LOG.info("水印检测：%s", diag)
    return mask, kept, diag


def _overlaps_any(c: WatermarkCandidate, others: Sequence[WatermarkCandidate],
                  thr: float) -> bool:
    """判断候选框是否与已有候选显著重叠（IoU 简化版）。"""
    ax0, ay0, ax1, ay1 = c.bbox
    aa = max(1, (ax1 - ax0) * (ay1 - ay0))
    for o in others:
        bx0, by0, bx1, by1 = o.bbox
        ix = max(0, min(ax1, bx1) - max(ax0, bx0))
        iy = max(0, min(ay1, by1) - max(ay0, by0))
        if ix * iy / aa > thr:
            return True
    return False


def refine_mask_for_watermark(image_rgb: np.ndarray, rough_mask: np.ndarray,
                              pad: int = 2) -> np.ndarray:
    """在粗框内进一步收紧到"笔画级"掩膜。

    为什么值得多做这一步？擦除区域越小，LaMa 需要"编造"的内容就越少，
    保留的原始像素就越多，结果自然越真实。整框擦除会把水印覆盖下的
    真实纹理也一并毁掉。
    """
    if rough_mask.max() == 0:
        return rough_mask
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    resp = cv2.max(cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k),
                   cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k))
    inside = (rough_mask > 0)
    if inside.sum() < 32:
        return rough_mask

    vals = resp[inside]
    thr = float(np.percentile(vals, 62))
    fine = np.zeros_like(rough_mask)
    fine[inside & (resp >= max(thr, 3))] = 255

    # 适度膨胀，确保覆盖抗锯齿边缘
    if pad > 0:
        fine = cv2.dilate(fine, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1)))
    # 若收得太狠（几乎什么都没剩），回退原框
    if fine.sum() < rough_mask.sum() * 0.04:
        return rough_mask
    return fine
