# -*- coding: utf-8 -*-
"""
models/ocr.py —— OCR 引擎统一封装
================================================================================
为什么要做"统一封装"？
----------------------
PaddleOCR 与 EasyOCR 的返回结构完全不同（前者是嵌套 list，后者是 tuple），
且不同版本的 PaddleOCR（2.x / 3.x）API 还变过好几次。如果让业务代码
直接对接它们，任何一次升级都会引发连锁修改。

这里把两者都归一化成 ``List[TextBox]``（见 utils.TextBox），
业务层只认这一个数据结构。

优先级
------
1. **PaddleOCR**：中文识别精度最好，det+rec 一体，返回的是**四点多边形**，
   能直接拿来做透视估计——这对"保持原文字倾斜/透视"至关重要。
2. **EasyOCR**：安装更轻，中文稍弱，同样返回四点。
3. 都没有 → 返回空列表，UI 提示用户改用手动框选。
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from utils import LOG, TextBox, order_quad


class OcrEngine:
    """OCR 统一入口（懒加载 + 自动降级）。"""

    def __init__(self, lang: str = "ch", prefer: str = "auto", use_gpu: bool = False):
        """
        参数
        ----
        lang    : 'ch' 中英混合 / 'en' 纯英文
        prefer  : 'auto' | 'paddle' | 'easy'
        use_gpu : 是否尝试用 GPU（无 GPU 会自动回落 CPU）
        """
        self.lang = lang
        self.prefer = prefer
        self.use_gpu = use_gpu
        self.backend: Optional[str] = None
        self._engine = None

    # ------------------------------------------------------------------ #
    @staticmethod
    def probe() -> str:
        """探测本机可用的 OCR 后端，返回可读描述。"""
        found = []
        try:
            import paddleocr  # noqa: F401
            found.append("PaddleOCR")
        except Exception:
            pass
        try:
            import easyocr  # noqa: F401
            found.append("EasyOCR")
        except Exception:
            pass
        return " / ".join(found) if found else "未安装（文字功能需手动框选）"

    # ------------------------------------------------------------------ #
    def load(self) -> Optional[str]:
        """加载 OCR 后端，返回后端名；都不可用返回 None。"""
        if self._engine is not None:
            return self.backend

        order = ["paddle", "easy"] if self.prefer in ("auto", "paddle") else ["easy", "paddle"]
        for b in order:
            if b == "paddle" and self._load_paddle():
                return self.backend
            if b == "easy" and self._load_easy():
                return self.backend
        LOG.warning("未找到可用 OCR 引擎，文字相关功能需手动框选区域")
        return None

    def _load_paddle(self) -> bool:
        try:
            from paddleocr import PaddleOCR
        except Exception:
            return False
        # 不同版本 PaddleOCR 参数名不一致，逐个尝试
        kwarg_sets = [
            dict(use_angle_cls=True, lang=self.lang, show_log=False,
                 use_gpu=self.use_gpu),
            dict(use_angle_cls=True, lang=self.lang, use_gpu=self.use_gpu),
            dict(use_textline_orientation=True, lang=self.lang),
            dict(lang=self.lang),
        ]
        for kw in kwarg_sets:
            try:
                self._engine = PaddleOCR(**kw)
                self.backend = "paddle"
                LOG.info("OCR 后端：PaddleOCR (%s)", kw)
                return True
            except Exception as e:
                last = e
        LOG.warning("PaddleOCR 初始化失败：%s", last)
        return False

    def _load_easy(self) -> bool:
        try:
            import easyocr
        except Exception:
            return False
        try:
            langs = ["ch_sim", "en"] if self.lang == "ch" else ["en"]
            self._engine = easyocr.Reader(langs, gpu=self.use_gpu, verbose=False)
            self.backend = "easy"
            LOG.info("OCR 后端：EasyOCR")
            return True
        except Exception as e:
            LOG.warning("EasyOCR 初始化失败：%s", e)
            return False

    # ------------------------------------------------------------------ #
    def detect(self, image_rgb: np.ndarray, min_score: float = 0.35) -> List[TextBox]:
        """检测并识别图像中的文字，返回归一化后的 TextBox 列表。"""
        if self.load() is None:
            return []
        try:
            if self.backend == "paddle":
                return self._detect_paddle(image_rgb, min_score)
            return self._detect_easy(image_rgb, min_score)
        except Exception as e:
            LOG.error("OCR 推理失败：%s", e)
            return []

    # -------------------------- 各后端解析 ---------------------------- #
    def _detect_paddle(self, img: np.ndarray, min_score: float) -> List[TextBox]:
        """兼容 PaddleOCR 2.x（ocr()）与 3.x（predict()）两种返回结构。"""
        raw = None
        for call in ("ocr", "predict"):
            fn = getattr(self._engine, call, None)
            if fn is None:
                continue
            try:
                raw = fn(img, cls=True) if call == "ocr" else fn(img)
                break
            except TypeError:
                try:
                    raw = fn(img)
                    break
                except Exception:
                    continue
            except Exception:
                continue
        if raw is None:
            return []

        boxes: List[TextBox] = []

        # ---- 结构 A：2.x → [[ [quad, (text, score)], ... ]]
        try:
            page = raw[0] if (isinstance(raw, (list, tuple)) and len(raw) == 1
                              and isinstance(raw[0], (list, tuple))) else raw
            for item in page:
                if not (isinstance(item, (list, tuple)) and len(item) >= 2):
                    raise TypeError
                quad, info = item[0], item[1]
                text, score = (info[0], float(info[1])) if isinstance(info, (list, tuple)) \
                    else (str(info), 1.0)
                if score < min_score or not str(text).strip():
                    continue
                boxes.append(TextBox(order_quad(np.array(quad, dtype=np.float32)),
                                     str(text), score))
            if boxes:
                return boxes
        except Exception:
            boxes = []

        # ---- 结构 B：3.x → [{'dt_polys': [...], 'rec_texts': [...], 'rec_scores': [...]}]
        try:
            d = raw[0] if isinstance(raw, (list, tuple)) else raw
            if hasattr(d, "get") or isinstance(d, dict):
                polys = d.get("dt_polys") or d.get("rec_polys") or []
                texts = d.get("rec_texts") or []
                scores = d.get("rec_scores") or [1.0] * len(texts)
                for q, t, s in zip(polys, texts, scores):
                    if float(s) < min_score or not str(t).strip():
                        continue
                    boxes.append(TextBox(order_quad(np.array(q, dtype=np.float32)),
                                         str(t), float(s)))
        except Exception as e:
            LOG.warning("PaddleOCR 结果解析失败：%s", e)
        return boxes

    def _detect_easy(self, img: np.ndarray, min_score: float) -> List[TextBox]:
        res = self._engine.readtext(img)
        out = []
        for quad, text, score in res:
            if float(score) < min_score or not str(text).strip():
                continue
            out.append(TextBox(order_quad(np.array(quad, dtype=np.float32)),
                               str(text), float(score)))
        return out
