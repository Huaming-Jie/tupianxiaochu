# -*- coding: utf-8 -*-
"""
core/pdf_io.py —— PDF 高保真互转
================================================================================
需求里写着"**保留原清晰度**"，这句话在 PDF 场景下有个陷阱：

PDF 是矢量容器，页面尺寸单位是 **点（pt，1pt = 1/72 英寸）**，而不是像素。
如果你固定用 150 DPI 去光栅化，一份原本内嵌了 600 DPI 扫描图的合同，
会被你**降采样到四分之一**——文字边缘立刻发虚。

所以这里的做法是 **按页自适应 DPI**：

1. 扫描该页所有内嵌位图，取最大者的像素宽度 ``px_w``；
2. 该页宽度为 ``pt_w`` 点，则页面原生分辨率 ``native_dpi = px_w / (pt_w/72)``；
3. 实际渲染 DPI = ``clamp(max(用户设定, native_dpi), 72, 上限)``。

这样：扫描件按其原生分辨率还原（不丢一个像素），纯矢量页按用户设定
渲染（通常 300 DPI 足够印刷）。

回写时，页面尺寸严格沿用原页的 pt 尺寸，图片以原像素数嵌入，
因此在任何阅读器里打开都与原件同尺寸、同清晰度。
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from utils import LOG

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except Exception:                                    # pragma: no cover
    fitz = None
    HAS_FITZ = False


@dataclass
class PdfPage:
    """一页 PDF 的渲染结果与还原所需的元信息。"""
    index: int
    image: np.ndarray                # RGB uint8
    width_pt: float                  # 原页宽（点）
    height_pt: float                 # 原页高（点）
    dpi: float                       # 实际渲染 DPI
    rotation: int = 0

    @property
    def size_px(self) -> Tuple[int, int]:
        return self.image.shape[1], self.image.shape[0]


# --------------------------------------------------------------------------- #
def _require_fitz():
    if not HAS_FITZ:
        raise RuntimeError(
            "未安装 PyMuPDF，无法处理 PDF。请执行：pip install PyMuPDF")


def page_native_dpi(page) -> float:
    """估算某一页内嵌位图的原生 DPI；无位图时返回 0。"""
    try:
        rect = page.rect
        pt_w = float(rect.width) or 1.0
        best_px = 0
        for info in page.get_images(full=True):
            # info: (xref, smask, width, height, bpc, colorspace, ...)
            w_px = int(info[2])
            best_px = max(best_px, w_px)
        if best_px <= 0:
            return 0.0
        return best_px / (pt_w / 72.0)
    except Exception:
        return 0.0


# --------------------------------------------------------------------------- #
def pdf_to_pages(path: str, dpi: int = 300, max_dpi: int = 600,
                 adaptive: bool = True,
                 progress: Optional[Callable[[int, int], None]] = None
                 ) -> List[PdfPage]:
    """把 PDF 光栅化成逐页 RGB 图像。

    参数
    ----
    dpi      : 基准 DPI（纯矢量页使用）
    max_dpi  : 上限，防止 1200 DPI 的巨幅扫描件把内存撑爆
    adaptive : 是否启用"按页自适应 DPI"
    """
    _require_fitz()
    doc = fitz.open(path)
    pages: List[PdfPage] = []
    n = doc.page_count

    for i in range(n):
        page = doc.load_page(i)
        use_dpi = float(dpi)
        if adaptive:
            nd = page_native_dpi(page)
            if nd > 0:
                use_dpi = max(use_dpi, nd)
        use_dpi = float(min(max(use_dpi, 72.0), max_dpi))

        zoom = use_dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n)
        if pix.n == 4:
            # 透明区域按白底合成（alpha=False 时通常已合成，这里做兜底）
            rgb = img[:, :, :3].astype(np.float32)
            a = img[:, :, 3:4].astype(np.float32) / 255.0
            img = (rgb * a + 255.0 * (1.0 - a)).astype(np.uint8)
        elif pix.n == 1:
            img = np.repeat(img, 3, axis=2)

        pages.append(PdfPage(
            index=i, image=np.ascontiguousarray(img),
            width_pt=float(page.rect.width),
            height_pt=float(page.rect.height),
            dpi=use_dpi, rotation=int(page.rotation),
        ))
        LOG.info("PDF 第 %d/%d 页：%dx%d @ %.0f DPI",
                 i + 1, n, pix.width, pix.height, use_dpi)
        if progress:
            progress(i + 1, n)

    doc.close()
    return pages


# --------------------------------------------------------------------------- #
def pages_to_pdf(pages: Sequence[PdfPage], out_path: str,
                 lossless: bool = True, jpeg_quality: int = 95,
                 progress: Optional[Callable[[int, int], None]] = None) -> str:
    """把（可能已被编辑的）页面图像重新合成为 PDF。

    参数
    ----
    lossless : True → 内嵌 PNG（无损，体积大）；False → 内嵌高质量 JPEG
               扫描件建议用 JPEG（体积可小 5~10 倍且肉眼无差）；
               含大片纯色/线稿的页面建议 PNG。
    """
    _require_fitz()
    doc = fitz.open()
    n = len(pages)

    for i, p in enumerate(pages):
        buf = io.BytesIO()
        pil = Image.fromarray(p.image)
        if lossless:
            pil.save(buf, format="PNG", compress_level=4)
        else:
            pil.convert("RGB").save(buf, format="JPEG",
                                    quality=jpeg_quality, subsampling=0)
        data = buf.getvalue()

        page = doc.new_page(width=p.width_pt, height=p.height_pt)
        page.insert_image(fitz.Rect(0, 0, p.width_pt, p.height_pt),
                          stream=data, keep_proportion=False)
        if progress:
            progress(i + 1, n)

    doc.save(out_path, garbage=3, deflate=True)
    doc.close()
    LOG.info("PDF 已导出：%s（%d 页）", out_path, n)
    return out_path


# --------------------------------------------------------------------------- #
def pages_to_pdf_preserved(src_pdf_path: str, pages: Sequence[PdfPage],
                           edited_indices: Sequence[int], out_path: str,
                           lossless: bool = True, jpeg_quality: int = 95,
                           progress: Optional[Callable[[int, int], None]] = None
                           ) -> str:
    """重建 PDF，但**未编辑的页直接从原文件逐页拷贝**（字节级保真），
    只有 ``edited_indices`` 里的页才用（可能已被编辑的）光栅图重嵌。

    价值：多页 PDF 只改其中一两页时，其余页零损失、零重编码，
    文件也更小；且矢量/文字层、注释、超链接等元数据都原样保留。
    若原文件不可用或出错，调用方应回退到 :func:`pages_to_pdf` 全量光栅化。
    """
    _require_fitz()
    edited = set(int(i) for i in edited_indices)
    src = fitz.open(src_pdf_path)
    out = fitz.open()
    n = len(pages)

    for i in range(n):
        if i not in edited:
            # 原样拷贝该页（保留矢量/文字层、注释、超链接等）
            out.insert_pdf(src, from_page=i, to_page=i)
        else:
            p = pages[i]
            buf = io.BytesIO()
            pil = Image.fromarray(p.image)
            if lossless:
                pil.save(buf, format="PNG", compress_level=4)
            else:
                pil.convert("RGB").save(buf, format="JPEG",
                                        quality=jpeg_quality, subsampling=0)
            data = buf.getvalue()
            page = out.new_page(width=p.width_pt, height=p.height_pt)
            page.insert_image(fitz.Rect(0, 0, p.width_pt, p.height_pt),
                              stream=data, keep_proportion=False)
        if progress:
            progress(i + 1, n)

    out.save(out_path, garbage=3, deflate=True)
    out.close()
    src.close()
    LOG.info("PDF 已导出（保留未改页）：%s（%d 页，编辑 %d 页）",
             out_path, n, len(edited))
    return out_path


# --------------------------------------------------------------------------- #
def images_to_pdf(images: Sequence[np.ndarray], out_path: str,
                  dpi: int = 300, lossless: bool = True,
                  jpeg_quality: int = 95) -> str:
    """把一组普通图片合成 PDF（每张一页，按 DPI 换算页面尺寸）。"""
    pages = []
    for i, im in enumerate(images):
        h, w = im.shape[:2]
        pages.append(PdfPage(index=i, image=im,
                             width_pt=w * 72.0 / dpi,
                             height_pt=h * 72.0 / dpi, dpi=float(dpi)))
    return pages_to_pdf(pages, out_path, lossless, jpeg_quality)


def probe_pdf(path: str) -> str:
    """返回 PDF 的概要信息（页数 / 尺寸 / 原生 DPI），用于 UI 提示。"""
    _require_fitz()
    doc = fitz.open(path)
    lines = [f"共 {doc.page_count} 页"]
    for i in range(min(3, doc.page_count)):
        pg = doc.load_page(i)
        nd = page_native_dpi(pg)
        lines.append(f"  第{i + 1}页 {pg.rect.width:.0f}×{pg.rect.height:.0f}pt"
                     f"，原生≈{nd:.0f}DPI" if nd else
                     f"  第{i + 1}页 {pg.rect.width:.0f}×{pg.rect.height:.0f}pt（矢量）")
    doc.close()
    return "\n".join(lines)
