# -*- coding: utf-8 -*-
"""
utils.py —— 通用工具层
================================================================================
本模块不依赖任何深度学习框架，是整个项目最底层的"地基"，提供：

1. 日志与路径工具
2. 图像格式互转（numpy / PIL / 字节流）
3. 掩膜（mask）处理：羽化、膨胀、二值化、包围盒
4. ROI（感兴趣区域）裁切与"按掩膜无损回贴"——保证掩膜外像素逐字节不变
5. 几何工具：四边形排序、透视变换、角度估计
6. 色彩工具：主色聚类（区分文字前景色 / 背景色）、光照匹配
7. 字体发现：跨平台查找可渲染中文的字体文件

设计要点
--------
* **无损回贴**是本项目质量的生命线。任何修复算法都只在 ROI 内运行，
  再用羽化后的 alpha 掩膜与原图做加权合成，掩膜为 0 的区域直接取原图，
  因此"人物、景物绝对不受损"这一硬性要求在数学上成立。
* 所有函数统一使用 **RGB uint8 HxWx3** 作为图像约定（OpenCV 默认 BGR，
  进出边界处显式转换），避免全项目色彩通道混乱。
"""

from __future__ import annotations

import logging
import os
import platform
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------- #
# 0. 全局常量
# --------------------------------------------------------------------------- #

#: 项目根目录。PyInstaller 冻结后 __file__ 指向临时解包路径，
#: 因此用 sys.executable 所在目录作为稳定根目录。
if getattr(sys, "frozen", False):
    PROJECT_ROOT = os.path.dirname(os.path.abspath(sys.executable))
else:
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

#: 模型权重缓存目录，可用环境变量 SIE_MODEL_DIR 覆盖。
#: 冻结后固定放到用户目录（%APPDATA% 或 ~），避免 onefile 临时目录被清除导致模型丢失。
if getattr(sys, "frozen", False):
    _appdata = os.environ.get("APPDATA") or os.path.expanduser("~")
    MODEL_DIR = os.path.join(_appdata, "smart_image_editor", "models")
else:
    MODEL_DIR = os.environ.get("SIE_MODEL_DIR", os.path.join(PROJECT_ROOT, "models_zoo"))

#: 支持的图片扩展名
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")

#: 支持的全部可打开扩展名
ALL_EXTS = IMAGE_EXTS + (".pdf",)


# --------------------------------------------------------------------------- #
# 1. 日志
# --------------------------------------------------------------------------- #

def setup_logger(name: str = "SIE", level: int = logging.INFO) -> logging.Logger:
    """创建（或复用）一个带统一格式的 logger。

    多次调用不会重复添加 handler，避免日志重复打印。
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
                          datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


LOG = setup_logger()


def ensure_dir(path: str) -> str:
    """确保目录存在并返回该路径。"""
    os.makedirs(path, exist_ok=True)
    return path


# --------------------------------------------------------------------------- #
# 2. 图像 IO 与格式转换
# --------------------------------------------------------------------------- #

def imread_rgb(path: str) -> np.ndarray:
    """读取任意常见格式图片为 RGB uint8 ndarray。

    使用 ``np.fromfile`` + ``cv2.imdecode``，规避 OpenCV 在 Windows 下
    无法处理中文路径的经典坑；WEBP / TIFF 交给 Pillow 兜底。
    """
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    except Exception:
        img = None

    if img is None:  # OpenCV 解码失败 → Pillow 兜底
        with Image.open(path) as im:
            return np.array(im.convert("RGB"))

    if img.ndim == 2:                       # 灰度
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:                   # 带 alpha：与白底合成
        bgr = img[:, :, :3].astype(np.float32)
        a = (img[:, :, 3:4].astype(np.float32)) / 255.0
        bgr = bgr * a + 255.0 * (1 - a)
        img = bgr.astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def imread_rgba(path: str) -> np.ndarray:
    """读取图片为 RGBA（Logo 导入时需要保留透明通道）。"""
    with Image.open(path) as im:
        return np.array(im.convert("RGBA"))


def imwrite_rgb(path: str, img_rgb: np.ndarray, quality: int = 100) -> None:
    """保存 RGB ndarray 到磁盘，保持原分辨率与最高画质。

    * PNG / WEBP 使用无损参数
    * JPEG 使用 quality=100 且关闭色度二次采样（4:4:4），最大限度保留色彩深度
    """
    ext = os.path.splitext(path)[1].lower()
    pil = Image.fromarray(img_rgb)
    if ext in (".jpg", ".jpeg"):
        pil.save(path, quality=quality, subsampling=0, optimize=True)
    elif ext == ".webp":
        pil.save(path, lossless=True, quality=100)
    elif ext in (".tif", ".tiff"):
        pil.save(path, compression="tiff_lzw")
    else:
        pil.save(path, compress_level=3)


def pil_to_np(pil_img: Image.Image) -> np.ndarray:
    """PIL → RGB ndarray。"""
    return np.array(pil_img.convert("RGB"))


def np_to_pil(arr: np.ndarray) -> Image.Image:
    """RGB ndarray → PIL。"""
    return Image.fromarray(arr)


def to_bgr(img_rgb: np.ndarray) -> np.ndarray:
    """RGB → BGR（调用 OpenCV 专有算法时使用）。"""
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    """BGR → RGB。"""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- #
# 3. 掩膜工具
# --------------------------------------------------------------------------- #

def binarize_mask(mask: np.ndarray, thr: int = 127) -> np.ndarray:
    """任意灰度/彩色掩膜 → 0/255 单通道 uint8。"""
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    return ((mask > thr).astype(np.uint8)) * 255


def dilate_mask(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    """膨胀掩膜。

    为什么修复前几乎总要膨胀？因为待擦除物体的**抗锯齿边缘**会溢出用户
    涂抹的范围，若不外扩，残留的半透明轮廓会形成"鬼影"。
    """
    if radius <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    return cv2.dilate(mask, k, iterations=1)


def feather_mask(mask: np.ndarray, radius: int = 4) -> np.ndarray:
    """把 0/255 硬掩膜转成 0.0~1.0 的软 alpha，用于无缝合成。

    做法：先高斯模糊，再做一次 gamma 校正让过渡更贴近人眼感知。
    """
    if radius <= 0:
        return (mask.astype(np.float32) / 255.0)
    k = radius * 2 + 1
    soft = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (k, k), radius / 2.0)
    return np.clip(soft, 0.0, 1.0)


def mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """返回掩膜非零区域的包围盒 (x0, y0, x1, y1)，全零时返回 None。"""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def expand_bbox(bbox: Tuple[int, int, int, int], pad: int,
                w: int, h: int) -> Tuple[int, int, int, int]:
    """按 pad 像素外扩包围盒并裁剪到图像范围内。

    外扩的意义：修复模型需要"上下文"才能推断被遮挡处应该长什么样，
    只喂一个紧贴目标的小框，模型无从参考周围纹理。
    """
    x0, y0, x1, y1 = bbox
    return (max(0, x0 - pad), max(0, y0 - pad),
            min(w, x1 + pad), min(h, y1 + pad))


def quad_to_mask(shape: Tuple[int, int], quad: Sequence[Sequence[float]]) -> np.ndarray:
    """由四边形顶点生成 0/255 掩膜。"""
    m = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(m, [np.asarray(quad, dtype=np.int32)], 255)
    return m


def pad_to_multiple(img: np.ndarray, mod: int = 8) -> Tuple[np.ndarray, Tuple[int, int]]:
    """把图像右/下侧反射填充到 mod 的整数倍，返回 (新图, (原高, 原宽))。

    LaMa 等全卷积网络要求输入尺寸能被 8 整除。用 **反射填充** 而非补零，
    可避免边界处出现黑边伪影。
    """
    h, w = img.shape[:2]
    nh = (h + mod - 1) // mod * mod
    nw = (w + mod - 1) // mod * mod
    if nh == h and nw == w:
        return img, (h, w)
    pad = ((0, nh - h), (0, nw - w)) + (((0, 0),) if img.ndim == 3 else ())
    return np.pad(img, pad, mode="reflect" if min(h, w) > 1 else "edge"), (h, w)


def unpad(img: np.ndarray, orig_hw: Tuple[int, int]) -> np.ndarray:
    """撤销 pad_to_multiple 的填充。"""
    h, w = orig_hw
    return img[:h, :w]


# --------------------------------------------------------------------------- #
# 4. ROI 裁切 / 无损回贴 —— 全项目质量保证的核心
# --------------------------------------------------------------------------- #

@dataclass
class RoiBox:
    """一个 ROI 区域的描述。"""
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.x0, self.y0, self.x1, self.y1


def compute_roi(mask: np.ndarray, context_ratio: float = 0.6,
                min_pad: int = 48, max_side: int = 2048) -> Optional[RoiBox]:
    """根据掩膜计算带上下文的 ROI。

    参数
    ----
    context_ratio : 以掩膜短边为基准向外扩展的比例，越大上下文越丰富、越慢
    min_pad       : 最小外扩像素，防止极小目标缺乏上下文
    max_side      : ROI 最长边上限，避免超大图 OOM（超出时仍保留完整掩膜）
    """
    bbox = mask_bbox(mask)
    if bbox is None:
        return None
    h, w = mask.shape[:2]
    x0, y0, x1, y1 = bbox
    base = min(x1 - x0, y1 - y0)
    pad = int(max(min_pad, base * context_ratio))
    x0, y0, x1, y1 = expand_bbox((x0, y0, x1, y1), pad, w, h)

    # 若 ROI 过大，向内收缩上下文（但绝不裁掉掩膜本身）
    if max(x1 - x0, y1 - y0) > max_side:
        bx0, by0, bx1, by1 = bbox
        cx, cy = (bx0 + bx1) // 2, (by0 + by1) // 2
        half = max_side // 2
        x0 = max(0, min(bx0, cx - half))
        y0 = max(0, min(by0, cy - half))
        x1 = min(w, max(bx1, cx + half))
        y1 = min(h, max(by1, cy + half))
    return RoiBox(x0, y0, x1, y1)


def paste_back(base_rgb: np.ndarray, patch_rgb: np.ndarray, roi: RoiBox,
               mask_full: np.ndarray, feather: int = 4) -> np.ndarray:
    """把修复后的 ROI 补丁按羽化掩膜合成回原图。

    **关键保证**：alpha 为 0 的位置结果 = 原图像素（float 运算后再取整，
    由于 alpha 严格为 0，round(orig*1.0) == orig，不会产生任何偏差）。
    """
    out = base_rgb.copy()
    x0, y0, x1, y1 = roi.as_tuple()

    sub_mask = mask_full[y0:y1, x0:x1]
    alpha = feather_mask(sub_mask, feather)[..., None]           # (h,w,1) 0~1

    orig_sub = base_rgb[y0:y1, x0:x1].astype(np.float32)
    new_sub = patch_rgb.astype(np.float32)
    blended = new_sub * alpha + orig_sub * (1.0 - alpha)

    # 严格保证 alpha==0 的像素逐字节等于原图
    hard_keep = (alpha[..., 0] <= 1e-6)
    blended[hard_keep] = orig_sub[hard_keep]

    out[y0:y1, x0:x1] = np.clip(blended + 0.5, 0, 255).astype(np.uint8)
    return out


# --------------------------------------------------------------------------- #
# 5. 几何工具
# --------------------------------------------------------------------------- #

def order_quad(pts: Sequence[Sequence[float]]) -> np.ndarray:
    """把任意顺序的 4 个点排成 [左上, 右上, 右下, 左下]。

    思路：x+y 最小者为左上、最大者为右下；x-y 最小者为左下、最大者为右上。
    """
    p = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = p.sum(axis=1)
    d = (p[:, 0] - p[:, 1])
    tl = p[np.argmin(s)]
    br = p[np.argmax(s)]
    tr = p[np.argmax(d)]
    bl = p[np.argmin(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def quad_size(quad: np.ndarray) -> Tuple[float, float]:
    """返回四边形的"展平"宽高（上下边均值 / 左右边均值）。"""
    tl, tr, br, bl = quad
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
    return float(w), float(h)


def quad_angle(quad: np.ndarray) -> float:
    """由上边缘估计文本行的倾斜角（度，逆时针为正）。"""
    tl, tr = quad[0], quad[1]
    dx, dy = (tr - tl)
    return float(np.degrees(np.arctan2(dy, dx)))


def warp_rgba_to_quad(rgba: np.ndarray, quad: np.ndarray,
                      out_shape: Tuple[int, int]) -> np.ndarray:
    """把一张 RGBA 图透视变换到目标四边形，输出与画布同尺寸的 RGBA。

    这是"Logo 自动适配透视"与"文字透视变形"的共同底座。
    """
    h, w = rgba.shape[:2]
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, order_quad(quad).astype(np.float32))
    return cv2.warpPerspective(
        rgba, M, (out_shape[1], out_shape[0]),
        flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def alpha_composite(base_rgb: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
    """把 RGBA 前景合成到 RGB 背景上（标准 source-over）。"""
    a = overlay_rgba[:, :, 3:4].astype(np.float32) / 255.0
    fg = overlay_rgba[:, :, :3].astype(np.float32)
    bg = base_rgb.astype(np.float32)
    out = fg * a + bg * (1 - a)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# 6. 色彩与光照
# --------------------------------------------------------------------------- #

def dominant_two_colors(roi_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """对 ROI 做 2 类 K-Means，返回 (前景色, 背景色, 前景占比)。

    约定：**像素数较少的一类视为前景（文字笔画）**，较多的一类为背景。
    这是从图片里"偷"出原文字颜色的最稳做法——比取平均值准得多，
    因为平均值会被背景稀释成一个灰扑扑的中间色。
    """
    data = roi_rgb.reshape(-1, 3).astype(np.float32)
    if len(data) < 8:
        m = data.mean(axis=0)
        return m, m, 0.5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, label, center = cv2.kmeans(data, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    label = label.flatten()
    counts = np.bincount(label, minlength=2)
    fg_idx = int(np.argmin(counts))
    bg_idx = 1 - fg_idx
    ratio = float(counts[fg_idx]) / float(counts.sum())
    return center[fg_idx], center[bg_idx], ratio


def match_illumination(src_rgb: np.ndarray, dst_rgb: np.ndarray,
                       src_alpha: Optional[np.ndarray] = None,
                       strength: float = 0.85) -> np.ndarray:
    """把 src 的亮度统计对齐到 dst（新 Logo / 新文字融入原图光照的关键）。

    在 **LAB 空间**只调整 L 通道的均值与标准差，保留原有色相，
    避免直接在 RGB 上做统计匹配导致的偏色。
    """
    src_lab = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    dst_lab = cv2.cvtColor(dst_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    if src_alpha is not None:
        sel = src_alpha > 16
        if sel.sum() < 16:
            return src_rgb
        s_l = src_lab[:, :, 0][sel]
    else:
        s_l = src_lab[:, :, 0].reshape(-1)

    d_l = dst_lab[:, :, 0].reshape(-1)
    s_mu, s_sd = float(s_l.mean()), float(s_l.std() + 1e-6)
    d_mu, d_sd = float(d_l.mean()), float(d_l.std() + 1e-6)

    gain = np.clip(d_sd / s_sd, 0.6, 1.6)
    l = src_lab[:, :, 0]
    l_new = (l - s_mu) * gain + d_mu
    src_lab[:, :, 0] = l * (1 - strength) + np.clip(l_new, 0, 255) * strength
    return cv2.cvtColor(src_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


def estimate_blur_sigma(gray: np.ndarray) -> float:
    """粗略估计图像锐度对应的高斯 sigma，用于让新贴入的元素"同样地虚"。

    原理：拉普拉斯方差越大越锐利。经验映射到 0~1.2 的 sigma 区间。
    """
    v = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if v <= 0:
        return 0.8
    sigma = float(np.clip(1.6 - 0.25 * np.log1p(v), 0.0, 1.2))
    return sigma


def add_grain(img_rgb: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    """叠加轻微高斯噪点，模拟原图的传感器噪声/压缩颗粒。

    没有这一步，贴上去的新元素会"过于干净"，人眼一眼就能看出是后期加的。
    """
    if sigma <= 0.05:
        return img_rgb
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, img_rgb.shape).astype(np.float32)
    return np.clip(img_rgb.astype(np.float32) + noise + 0.5, 0, 255).astype(np.uint8)


def estimate_noise_sigma(gray: np.ndarray) -> float:
    """用中值绝对偏差（MAD）估计图像噪声标准差。"""
    h = cv2.Laplacian(gray, cv2.CV_64F)
    sigma = float(np.median(np.abs(h - np.median(h))) / 0.6745)
    return float(np.clip(sigma * 0.35, 0.0, 6.0))


# --------------------------------------------------------------------------- #
# 7. 字体发现与文本渲染
# --------------------------------------------------------------------------- #

#: 各平台常见的中文字体候选（按"通用度"排序）
_CJK_FONT_CANDIDATES = {
    "Windows": [
        r"C:\Windows\Fonts\msyh.ttc",      # 微软雅黑
        r"C:\Windows\Fonts\msyhbd.ttc",    # 微软雅黑 Bold
        r"C:\Windows\Fonts\simhei.ttf",    # 黑体
        r"C:\Windows\Fonts\simsun.ttc",    # 宋体
        r"C:\Windows\Fonts\simkai.ttf",    # 楷体
        r"C:\Windows\Fonts\Deng.ttf",      # 等线
    ],
    "Darwin": [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ],
    "Linux": [
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
    ],
}

_LATIN_FONT_CANDIDATES = {
    "Windows": [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf",
                r"C:\Windows\Fonts\times.ttf", r"C:\Windows\Fonts\calibri.ttf"],
    "Darwin": ["/Library/Fonts/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"],
    "Linux": ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
}


def list_available_fonts(cjk: bool = True) -> List[str]:
    """返回本机可用的字体文件路径列表（存在性已校验）。"""
    table = _CJK_FONT_CANDIDATES if cjk else _LATIN_FONT_CANDIDATES
    cands = table.get(platform.system(), []) + \
        table.get("Linux", []) + table.get("Windows", []) + table.get("Darwin", [])
    seen, out = set(), []
    for p in cands:
        if p not in seen and os.path.isfile(p):
            seen.add(p)
            out.append(p)

    # 兜底：扫描系统字体目录
    if not out:
        font_dirs = [r"C:\Windows\Fonts", "/usr/share/fonts", "/Library/Fonts",
                     os.path.expanduser("~/.fonts")]
        for d in font_dirs:
            if not os.path.isdir(d):
                continue
            for root, _, files in os.walk(d):
                for f in files:
                    if f.lower().endswith((".ttf", ".ttc", ".otf")):
                        out.append(os.path.join(root, f))
                if len(out) > 60:
                    break
            if out:
                break
    return out


def has_cjk(text: str) -> bool:
    """判断字符串是否包含中日韩字符。"""
    return any("\u4e00" <= ch <= "\u9fff" or "\u3000" <= ch <= "\u30ff" for ch in text)


def pick_font(text: str, preferred: Optional[str] = None) -> str:
    """为给定文本挑一个能正常渲染的字体路径。"""
    if preferred and os.path.isfile(preferred):
        return preferred
    fonts = list_available_fonts(cjk=has_cjk(text))
    if not fonts:
        fonts = list_available_fonts(cjk=True)
    if not fonts:
        raise RuntimeError("未在本机找到任何可用字体文件，请在设置中手动指定 .ttf/.ttc")
    return fonts[0]


def fit_font_size(text: str, font_path: str, target_w: float, target_h: float,
                  max_size: int = 512) -> Tuple[ImageFont.FreeTypeFont, int]:
    """二分搜索出让文本恰好塞进 (target_w, target_h) 的字号。

    为什么要二分而不是按 `target_h` 直接取字号？因为 TrueType 的
    em-box 与实际墨迹（ink box）不等，中英文差异尤其大，直接取值必然溢出或过小。
    """
    lo, hi, best = 4, max_size, 4
    best_font = ImageFont.truetype(font_path, 4)
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            f = ImageFont.truetype(font_path, mid)
        except Exception:
            break
        box = f.getbbox(text)
        w, h = box[2] - box[0], box[3] - box[1]
        if w <= target_w and h <= target_h:
            best, best_font = mid, f
            lo = mid + 1
        else:
            hi = mid - 1
    return best_font, best


def render_text_rgba(text: str, font_path: str, font_size: int,
                     color: Tuple[int, int, int], supersample: int = 4,
                     stroke_w: int = 0, letter_spacing: float = 0.0) -> np.ndarray:
    """渲染一段文字为紧贴墨迹的 RGBA 图。

    使用 **4 倍超采样 + LANCZOS 下采样**，让笔画边缘的抗锯齿质量接近
    原生排版效果，避免出现台阶状锯齿这一"一眼假"的破绽。
    """
    ss = max(1, supersample)
    font = ImageFont.truetype(font_path, font_size * ss)

    if letter_spacing and len(text) > 1:
        # 手工逐字排布以支持字距（PIL 原生不支持 letter-spacing）
        widths, height = [], 0
        for ch in text:
            b = font.getbbox(ch)
            widths.append(b[2] - b[0] if b[2] > b[0] else font.getlength(ch))
            height = max(height, b[3])
        total_w = int(sum(widths) + letter_spacing * ss * (len(text) - 1)) + 8 * ss
        total_h = int(font.getbbox(text)[3] + 8 * ss)
        canvas = Image.new("RGBA", (max(1, total_w), max(1, total_h)), (0, 0, 0, 0))
        d = ImageDraw.Draw(canvas)
        x = 4 * ss
        for ch, cw in zip(text, widths):
            d.text((x, 4 * ss), ch, font=font, fill=(*color, 255),
                   stroke_width=stroke_w * ss, stroke_fill=(*color, 255))
            x += cw + letter_spacing * ss
    else:
        box = font.getbbox(text)
        pad = 6 * ss
        w = max(1, box[2] - box[0] + pad * 2)
        h = max(1, box[3] - box[1] + pad * 2)
        canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(canvas)
        d.text((pad - box[0], pad - box[1]), text, font=font, fill=(*color, 255),
               stroke_width=stroke_w * ss, stroke_fill=(*color, 255))

    arr = np.array(canvas)
    # 裁掉四周全透明区域，得到紧贴墨迹的最小矩形
    ys, xs = np.where(arr[:, :, 3] > 0)
    if len(xs):
        arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    if ss > 1:
        pil = Image.fromarray(arr).resize(
            (max(1, arr.shape[1] // ss), max(1, arr.shape[0] // ss)),
            Image.LANCZOS)
        arr = np.array(pil)
    return arr


# --------------------------------------------------------------------------- #
# 8. 杂项
# --------------------------------------------------------------------------- #

@dataclass
class TextBox:
    """OCR 检测到的一段文字。"""
    quad: np.ndarray                     # (4,2) float32，顺序为 左上→右上→右下→左下
    text: str = ""
    score: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        x0 = int(np.floor(self.quad[:, 0].min()))
        y0 = int(np.floor(self.quad[:, 1].min()))
        x1 = int(np.ceil(self.quad[:, 0].max()))
        y1 = int(np.ceil(self.quad[:, 1].max()))
        return x0, y0, x1, y1

    @property
    def center(self) -> Tuple[float, float]:
        return float(self.quad[:, 0].mean()), float(self.quad[:, 1].mean())


def human_size(num_bytes: float) -> str:
    """字节数 → 人类可读字符串。"""
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024 or unit == "GB":
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f}GB"


def clamp(v, lo, hi):
    """区间钳制。"""
    return max(lo, min(hi, v))
