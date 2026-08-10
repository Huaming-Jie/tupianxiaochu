# -*- coding: utf-8 -*-
"""
ui.py —— PyQt5 图形界面
================================================================================
界面结构
--------
::

    ┌─────────────────────────────────────────────────────────────┐
    │ 工具栏：打开 保存 导出PDF │ 撤销 重做 │ 适应窗口 1:1 │ 对比原图 │
    ├──────────┬──────────────────────────────────┬───────────────┤
    │ 工具箱   │                                  │ 参数面板      │
    │ ○物体    │          画    布                │ (随工具切换)  │
    │ ○水印    │   涂抹 / 框选 / 四点拾取         │               │
    │ ○文字    │   滚轮缩放 · 中键平移            │               │
    │ ○改字    │                                  │               │
    │ ○Logo    │                                  │               │
    ├──────────┴──────────────────────────────────┴───────────────┤
    │ 状态栏：引擎状态 │ 坐标 │ 进度条                            │
    └─────────────────────────────────────────────────────────────┘

两个关键工程决策
----------------
1. **画布只渲染可见区域**：面对 6000×4000 的大图，若每次鼠标移动都把整图
   转成 QPixmap 再缩放，帧率会掉到个位数。这里在 ``paintEvent`` 里先算出
   当前视口对应的**图像坐标子矩形**，只裁切并缩放这一小块。
   代价是 O(视口像素数)，与原图尺寸无关。

2. **所有推理跑在 QThread**：模型下载、LaMa 推理、OCR 都可能耗时数十秒。
   放在主线程会让窗口"未响应"。这里用一个通用 ``Worker`` 线程包装任意
   可调用对象，并把进度回调转成 Qt 信号。
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import (QPoint, QPointF, QRect, QRectF, QSize, Qt, QThread,
                          pyqtSignal)
from PyQt5.QtGui import (QColor, QFont, QImage, QKeySequence, QPainter, QPen,
                         QPixmap, QPolygonF)
from PyQt5.QtWidgets import (QAbstractItemView, QAction, QApplication,
                             QCheckBox, QColorDialog, QComboBox, QDialog,
                             QDialogButtonBox, QDoubleSpinBox, QFileDialog,
                             QFormLayout, QFrame, QGroupBox, QHBoxLayout,
                             QLabel, QLineEdit, QListWidget, QListWidgetItem,
                             QMainWindow, QMessageBox, QProgressBar,
                             QPushButton, QRadioButton, QScrollArea, QSlider,
                             QSpinBox, QSplitter, QStackedWidget, QStatusBar,
                             QTextEdit, QToolBar, QVBoxLayout, QWidget)

from core.inpaint import InpaintOptions, InpaintService
from core.logo_edit import LogoOptions, place_logo, quad_from_bbox, replace_logo
from core.pdf_io import (HAS_FITZ, PdfPage, pages_to_pdf, pages_to_pdf_preserved,
                         pdf_to_pages, probe_pdf)
from core.text_edit import (TextStyle, erase_text, estimate_style,
                            render_new_text, replace_text)
from core.watermark import auto_detect, refine_mask_for_watermark
from models.ocr import OcrEngine
from utils import (ALL_EXTS, IMAGE_EXTS, LOG, TextBox, binarize_mask,
                   imread_rgb, imread_rgba, imwrite_rgb,
                   list_available_fonts, order_quad, quad_to_mask)

# --------------------------------------------------------------------------- #
# 0. 工具函数
# --------------------------------------------------------------------------- #

def np_to_qimage(arr: np.ndarray) -> QImage:
    """numpy → QImage（内部做 copy，避免悬垂指针导致花屏/崩溃）。"""
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    if arr.ndim == 2:
        return QImage(arr.data, w, h, w, QImage.Format_Grayscale8).copy()
    if arr.shape[2] == 3:
        return QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    return QImage(arr.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()


# --------------------------------------------------------------------------- #
# 1. 后台工作线程
# --------------------------------------------------------------------------- #

class Worker(QThread):
    """通用后台任务线程。

    把任意 ``fn(*args, progress=cb, **kw)`` 丢进来执行，
    过程中的进度通过 Qt 信号回到主线程刷新 UI。
    """
    sig_progress = pyqtSignal(float, str)
    sig_done = pyqtSignal(object)
    sig_failed = pyqtSignal(str)

    def __init__(self, fn: Callable, *args, **kwargs):
        super().__init__()
        self._fn, self._args, self._kwargs = fn, args, kwargs

    def _progress(self, p: float, text: str = ""):
        self.sig_progress.emit(float(p), str(text))

    def run(self):
        try:
            # 若目标函数显式声明了 progress 参数，就把进度回调注入进去
            code = getattr(self._fn, "__code__", None)
            if code is not None and "progress" in code.co_varnames[:code.co_argcount]:
                self._kwargs.setdefault("progress", self._progress)
            result = self._fn(*self._args, **self._kwargs)
            self.sig_done.emit(result)
        except Exception:
            self.sig_failed.emit(traceback.format_exc())


# --------------------------------------------------------------------------- #
# 2. 画布控件
# --------------------------------------------------------------------------- #

class Canvas(QWidget):
    """支持缩放/平移/涂抹/框选/四点拾取的图像画布。"""

    MODE_BRUSH = "brush"
    MODE_ERASE = "erase"
    MODE_RECT = "rect"
    MODE_QUAD = "quad"
    MODE_PAN = "pan"

    sig_mask_changed = pyqtSignal()
    sig_quad_done = pyqtSignal(object)     # np.ndarray (4,2)
    sig_rect_done = pyqtSignal(object)     # (x0,y0,x1,y1)
    sig_cursor = pyqtSignal(int, int)
    sig_box_clicked = pyqtSignal(int)      # 点击了第 n 个标注框

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(480, 360)

        self._image: Optional[np.ndarray] = None      # RGB
        self._orig: Optional[np.ndarray] = None       # 对比用原图
        self._mask: Optional[np.ndarray] = None       # uint8 0/255
        self._pix: Optional[QPixmap] = None

        self.scale = 1.0
        self.origin = QPointF(0.0, 0.0)               # 视口左上角对应的图像坐标
        self.mode = self.MODE_BRUSH
        self.brush = 24
        self.show_original = False

        self.boxes: List[np.ndarray] = []             # 标注四边形
        self.box_labels: List[str] = []
        self.box_selected: set = set()

        self._drawing = False
        self._panning = False
        self._last_pt: Optional[QPointF] = None
        self._pan_anchor: Optional[QPoint] = None
        self._rect_start: Optional[QPointF] = None
        self._rect_cur: Optional[QPointF] = None
        self._quad_pts: List[QPointF] = []
        self._cursor_pos: Optional[QPoint] = None

    # ------------------------------------------------------------------ #
    # 数据接口
    # ------------------------------------------------------------------ #
    def set_image(self, img: np.ndarray, keep_view: bool = False,
                  reset_mask: bool = True):
        """载入一张新图。"""
        self._image = img
        self._pix = QPixmap.fromImage(np_to_qimage(img))
        if reset_mask or self._mask is None or self._mask.shape[:2] != img.shape[:2]:
            self._mask = np.zeros(img.shape[:2], np.uint8)
        if not keep_view:
            self.fit_to_window()
        self.update()

    def set_original(self, img: Optional[np.ndarray]):
        self._orig = img

    @property
    def image(self) -> Optional[np.ndarray]:
        return self._image

    @property
    def mask(self) -> Optional[np.ndarray]:
        return self._mask

    def set_mask(self, m: Optional[np.ndarray]):
        if m is None or self._image is None:
            return
        self._mask = binarize_mask(m)
        self.sig_mask_changed.emit()
        self.update()

    def clear_mask(self):
        if self._image is not None:
            self._mask = np.zeros(self._image.shape[:2], np.uint8)
            self.sig_mask_changed.emit()
            self.update()

    def add_to_mask(self, m: np.ndarray):
        if self._mask is None:
            return
        self._mask = cv2.bitwise_or(self._mask, binarize_mask(m))
        self.sig_mask_changed.emit()
        self.update()

    def set_boxes(self, quads: List[np.ndarray], labels: Optional[List[str]] = None):
        self.boxes = [np.asarray(q, np.float32) for q in quads]
        self.box_labels = labels or [""] * len(self.boxes)
        self.box_selected = set()
        self.update()

    def clear_quad_picking(self):
        self._quad_pts = []
        self.update()

    # ------------------------------------------------------------------ #
    # 视图变换
    # ------------------------------------------------------------------ #
    def fit_to_window(self):
        if self._image is None:
            return
        h, w = self._image.shape[:2]
        W, H = max(1, self.width()), max(1, self.height())
        self.scale = min(W / w, H / h)
        self.origin = QPointF(-(W / self.scale - w) / 2.0,
                              -(H / self.scale - h) / 2.0)
        self.update()

    def zoom_1to1(self):
        if self._image is None:
            return
        c = self._widget_to_image(QPointF(self.width() / 2, self.height() / 2))
        self.scale = 1.0
        self.origin = QPointF(c.x() - self.width() / 2, c.y() - self.height() / 2)
        self.update()

    def _widget_to_image(self, p: QPointF) -> QPointF:
        return QPointF(p.x() / self.scale + self.origin.x(),
                       p.y() / self.scale + self.origin.y())

    def _image_to_widget(self, p) -> QPointF:
        x, y = (p.x(), p.y()) if isinstance(p, QPointF) else (float(p[0]), float(p[1]))
        return QPointF((x - self.origin.x()) * self.scale,
                       (y - self.origin.y()) * self.scale)

    # ------------------------------------------------------------------ #
    # 绘制
    # ------------------------------------------------------------------ #
    def paintEvent(self, ev):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(46, 48, 52))
        if self._image is None or self._pix is None:
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(self.rect(), Qt.AlignCenter,
                             "请打开一张图片或 PDF\n（工具栏 → 打开）")
            return

        h, w = self._image.shape[:2]
        W, H = self.width(), self.height()

        # ---- 计算可见的图像子矩形（这是保证大图流畅的关键） ----
        sx0 = max(0.0, self.origin.x())
        sy0 = max(0.0, self.origin.y())
        sx1 = min(float(w), self.origin.x() + W / self.scale)
        sy1 = min(float(h), self.origin.y() + H / self.scale)
        if sx1 <= sx0 or sy1 <= sy0:
            return

        src = QRectF(sx0, sy0, sx1 - sx0, sy1 - sy0)
        tl = self._image_to_widget(QPointF(sx0, sy0))
        br = self._image_to_widget(QPointF(sx1, sy1))
        dst = QRectF(tl, br)

        painter.setRenderHint(QPainter.SmoothPixmapTransform, self.scale < 1.0)

        # ---- 底图（对比模式下画原图） ----
        if self.show_original and self._orig is not None:
            pm = QPixmap.fromImage(np_to_qimage(self._orig))
            painter.drawPixmap(dst, pm, src)
        else:
            painter.drawPixmap(dst, self._pix, src)

        # ---- 掩膜半透明叠加（只处理可见区域） ----
        if self._mask is not None and self._mask.max() > 0:
            ix0, iy0 = int(sx0), int(sy0)
            ix1, iy1 = int(np.ceil(sx1)), int(np.ceil(sy1))
            sub = self._mask[iy0:iy1, ix0:ix1]
            if sub.size:
                tw = max(1, int(dst.width()))
                th = max(1, int(dst.height()))
                small = cv2.resize(sub, (tw, th), interpolation=cv2.INTER_NEAREST)
                rgba = np.zeros((th, tw, 4), np.uint8)
                rgba[:, :, 0] = 255           # 红色
                rgba[:, :, 1] = 60
                rgba[:, :, 2] = 60
                rgba[:, :, 3] = (small > 0) * 115
                painter.drawImage(dst.topLeft(), np_to_qimage(rgba))

        # ---- 标注框 ----
        for i, q in enumerate(self.boxes):
            sel = i in self.box_selected
            pen = QPen(QColor(0, 200, 255) if sel else QColor(255, 190, 0))
            pen.setWidth(2 if sel else 1)
            painter.setPen(pen)
            poly = QPolygonF([self._image_to_widget(p) for p in q])
            painter.drawPolygon(poly)
            if self.box_labels and i < len(self.box_labels) and self.box_labels[i]:
                p0 = self._image_to_widget(q[0])
                painter.setFont(QFont("Microsoft YaHei", 8))
                painter.drawText(QPointF(p0.x(), p0.y() - 3), self.box_labels[i])

        # ---- 正在拉的矩形 ----
        if self._rect_start and self._rect_cur:
            pen = QPen(QColor(0, 230, 120)); pen.setWidth(2); pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            a = self._image_to_widget(self._rect_start)
            b = self._image_to_widget(self._rect_cur)
            painter.drawRect(QRectF(a, b).normalized())

        # ---- 四点拾取 ----
        if self._quad_pts:
            pen = QPen(QColor(0, 230, 255)); pen.setWidth(2)
            painter.setPen(pen)
            pts = [self._image_to_widget(p) for p in self._quad_pts]
            for i, p in enumerate(pts):
                painter.drawEllipse(p, 4, 4)
                painter.drawText(QPointF(p.x() + 6, p.y() - 6), str(i + 1))
            if len(pts) > 1:
                painter.drawPolyline(QPolygonF(pts))

        # ---- 画刷光标 ----
        if self.mode in (self.MODE_BRUSH, self.MODE_ERASE) and self._cursor_pos:
            pen = QPen(QColor(255, 255, 255, 200)); pen.setWidth(1)
            painter.setPen(pen)
            r = self.brush * self.scale / 2.0
            painter.drawEllipse(QPointF(self._cursor_pos), r, r)

    # ------------------------------------------------------------------ #
    # 交互
    # ------------------------------------------------------------------ #
    def wheelEvent(self, ev):
        if self._image is None:
            return
        if ev.modifiers() & Qt.ShiftModifier:          # Shift+滚轮 = 调笔刷
            self.brush = int(np.clip(self.brush + (3 if ev.angleDelta().y() > 0 else -3),
                                     2, 400))
            self.update()
            return
        # 以光标为锚点缩放
        pos = QPointF(ev.pos())
        before = self._widget_to_image(pos)
        factor = 1.15 if ev.angleDelta().y() > 0 else 1 / 1.15
        self.scale = float(np.clip(self.scale * factor, 0.02, 32.0))
        after = self._widget_to_image(pos)
        self.origin += (before - after)
        self.update()

    def mousePressEvent(self, ev):
        if self._image is None:
            return
        ip = self._widget_to_image(QPointF(ev.pos()))

        if ev.button() == Qt.MiddleButton or self.mode == self.MODE_PAN:
            self._panning = True
            self._pan_anchor = ev.pos()
            self.setCursor(Qt.ClosedHandCursor)
            return

        if ev.button() == Qt.LeftButton:
            # 先判断是否点中了某个标注框
            for i, q in enumerate(self.boxes):
                if cv2.pointPolygonTest(q.astype(np.float32),
                                        (float(ip.x()), float(ip.y())), False) >= 0:
                    self.sig_box_clicked.emit(i)
                    break

            if self.mode in (self.MODE_BRUSH, self.MODE_ERASE):
                self._drawing = True
                self._last_pt = ip
                self._stroke(ip, ip)
            elif self.mode == self.MODE_RECT:
                self._rect_start = ip
                self._rect_cur = ip
            elif self.mode == self.MODE_QUAD:
                self._quad_pts.append(ip)
                if len(self._quad_pts) == 4:
                    q = np.array([[p.x(), p.y()] for p in self._quad_pts], np.float32)
                    self.sig_quad_done.emit(q)
                    self._quad_pts = []
                self.update()

    def mouseMoveEvent(self, ev):
        self._cursor_pos = ev.pos()
        if self._image is None:
            self.update()
            return
        ip = self._widget_to_image(QPointF(ev.pos()))
        self.sig_cursor.emit(int(ip.x()), int(ip.y()))

        if self._panning and self._pan_anchor is not None:
            d = ev.pos() - self._pan_anchor
            self.origin -= QPointF(d.x() / self.scale, d.y() / self.scale)
            self._pan_anchor = ev.pos()
            self.update()
            return

        if self._drawing and self._last_pt is not None:
            self._stroke(self._last_pt, ip)
            self._last_pt = ip
        elif self._rect_start is not None:
            self._rect_cur = ip
        self.update()

    def mouseReleaseEvent(self, ev):
        if self._panning:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            return
        if self._drawing:
            self._drawing = False
            self._last_pt = None
            self.sig_mask_changed.emit()
        if self._rect_start is not None and self._rect_cur is not None:
            a, b = self._rect_start, self._rect_cur
            x0, x1 = sorted([int(a.x()), int(b.x())])
            y0, y1 = sorted([int(a.y()), int(b.y())])
            self._rect_start = self._rect_cur = None
            if x1 - x0 > 2 and y1 - y0 > 2:
                self.sig_rect_done.emit((x0, y0, x1, y1))
            self.update()

    def leaveEvent(self, ev):
        self._cursor_pos = None
        self.update()

    def _stroke(self, p0: QPointF, p1: QPointF):
        """在掩膜上画一段线（涂抹/擦除）。"""
        if self._mask is None:
            return
        val = 255 if self.mode == self.MODE_BRUSH else 0
        cv2.line(self._mask,
                 (int(round(p0.x())), int(round(p0.y()))),
                 (int(round(p1.x())), int(round(p1.y()))),
                 val, max(1, int(self.brush)), lineType=cv2.LINE_8)


# --------------------------------------------------------------------------- #
# 3. 文字编辑对话框
# --------------------------------------------------------------------------- #

class TextEditDialog(QDialog):
    """修改单段文字的参数面板。"""

    def __init__(self, box: TextBox, style: TextStyle, parent=None):
        super().__init__(parent)
        self.setWindowTitle("修改文字")
        self.resize(460, 380)
        self.box, self.style = box, style
        self._color = QColor(*[int(c) for c in style.fg_color])

        lay = QVBoxLayout(self)

        info = QLabel(f"<b>原文字：</b>{box.text or '（未识别）'}<br>"
                      f"<b>估计风格：</b>{style.describe()}")
        info.setWordWrap(True)
        lay.addWidget(info)

        form = QFormLayout()
        self.ed_text = QLineEdit(box.text)
        form.addRow("新文字：", self.ed_text)

        self.cb_fit = QComboBox()
        self.cb_fit.addItem("填满原框（缩放字号）", "fit_box")
        self.cb_fit.addItem("保持原字号（框随文字延展）", "keep_size")
        form.addRow("排版方式：", self.cb_fit)

        row = QHBoxLayout()
        self.btn_color = QPushButton()
        self._refresh_color_btn()
        self.btn_color.clicked.connect(self._pick_color)
        self.chk_auto_color = QCheckBox("沿用原色")
        self.chk_auto_color.setChecked(True)
        row.addWidget(self.btn_color); row.addWidget(self.chk_auto_color)
        wrap = QWidget(); wrap.setLayout(row)
        form.addRow("文字颜色：", wrap)

        self.cb_font = QComboBox()
        self.cb_font.addItem("自动匹配", "")
        for f in list_available_fonts(True)[:40]:
            self.cb_font.addItem(os.path.basename(f), f)
        form.addRow("字体：", self.cb_font)

        self.sp_scale = QDoubleSpinBox()
        self.sp_scale.setRange(0.3, 3.0); self.sp_scale.setSingleStep(0.05)
        self.sp_scale.setValue(1.0)
        form.addRow("字号微调：", self.sp_scale)

        self.chk_tight = QCheckBox("笔画级擦除（保留更多背景，推荐）")
        self.chk_tight.setChecked(True)
        form.addRow("", self.chk_tight)

        lay.addLayout(form)

        tip = QLabel("提示：若原文字带描边或阴影，取消勾选「笔画级擦除」更稳妥。")
        tip.setStyleSheet("color:#888;")
        tip.setWordWrap(True)
        lay.addWidget(tip)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _refresh_color_btn(self):
        self.btn_color.setStyleSheet(
            f"background-color: {self._color.name()}; min-width:80px;")
        self.btn_color.setText(self._color.name().upper())

    def _pick_color(self):
        c = QColorDialog.getColor(self._color, self, "选择文字颜色")
        if c.isValid():
            self._color = c
            self.chk_auto_color.setChecked(False)
            self._refresh_color_btn()

    def values(self) -> dict:
        return dict(
            new_text=self.ed_text.text(),
            fit_mode=self.cb_fit.currentData(),
            color=None if self.chk_auto_color.isChecked()
            else (self._color.red(), self._color.green(), self._color.blue()),
            font=self.cb_font.currentData() or None,
            size_scale=float(self.sp_scale.value()),
            tight=self.chk_tight.isChecked(),
        )


# --------------------------------------------------------------------------- #
# 4. 主窗口
# --------------------------------------------------------------------------- #

class MainWindow(QMainWindow):
    """智能图像编辑器主窗口。"""

    def __init__(self, device: str = "auto"):
        super().__init__()
        self.setWindowTitle("智能图像编辑器 · Smart Image Editor")
        self.resize(1480, 940)

        # ---- 文档状态 ----
        self.pages: List[PdfPage] = []
        self.page_idx = 0
        self.src_path: Optional[str] = None
        self.is_pdf = False
        self.edited_pages: set = set()   # 记录被编辑过的页码，导出时保留未改页
        self._undo: List[np.ndarray] = []
        self._redo: List[np.ndarray] = []
        self._busy = False
        self._worker: Optional[Worker] = None

        # ---- 算法服务 ----
        self.service = InpaintService(device=device)
        self.ocr = OcrEngine()
        self.ocr_boxes: List[TextBox] = []
        self.wm_cands = []
        self.logo_rgba: Optional[np.ndarray] = None
        self.logo_quad: Optional[np.ndarray] = None

        self._build_ui()
        self._update_status("就绪 · " + self.service.describe())

    # ================================================================== #
    # UI 构建
    # ================================================================== #
    def _build_ui(self):
        self.canvas = Canvas()
        self.canvas.sig_cursor.connect(self._on_cursor)
        self.canvas.sig_rect_done.connect(self._on_rect)
        self.canvas.sig_quad_done.connect(self._on_quad)
        self.canvas.sig_box_clicked.connect(self._on_box_clicked)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self.canvas)
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([210, 940, 330])
        self.setCentralWidget(splitter)

        self._build_toolbar()
        self._build_statusbar()

        # 切换到默认工具（此时 stack 已创建完毕）
        self._switch_tool(0)

    # ------------------------------------------------------------------ #
    def _build_toolbar(self):
        tb = QToolBar("主工具栏")
        tb.setIconSize(QSize(18, 18))
        self.addToolBar(tb)

        def act(text, slot, shortcut=None, tip=""):
            a = QAction(text, self)
            a.triggered.connect(slot)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            a.setToolTip(tip or text)
            tb.addAction(a)
            return a

        act("打开", self.open_file, "Ctrl+O", "打开图片或 PDF")
        act("保存", self.save_file, "Ctrl+S", "另存为图片")
        act("导出PDF", self.export_pdf, "Ctrl+Shift+S", "把全部页面导出为 PDF")
        tb.addSeparator()
        act("撤销", self.undo, "Ctrl+Z")
        act("重做", self.redo, "Ctrl+Y")
        tb.addSeparator()
        act("适应窗口", lambda: self.canvas.fit_to_window(), "Ctrl+0")
        act("1:1", lambda: self.canvas.zoom_1to1(), "Ctrl+1")
        tb.addSeparator()

        self.chk_compare = QCheckBox("对比原图")
        self.chk_compare.setToolTip("勾选后画布显示本页最初的样子（快捷键 Tab）")
        self.chk_compare.stateChanged.connect(self._on_compare)
        tb.addWidget(self.chk_compare)

        a = QAction("切换对比", self)
        a.setShortcut(QKeySequence(Qt.Key_Tab))
        a.triggered.connect(lambda: self.chk_compare.setChecked(
            not self.chk_compare.isChecked()))
        self.addAction(a)

        tb.addSeparator()
        tb.addWidget(QLabel("  页："))
        self.cb_page = QComboBox()
        self.cb_page.setMinimumWidth(110)
        self.cb_page.currentIndexChanged.connect(self._on_page_change)
        tb.addWidget(self.cb_page)

    # ------------------------------------------------------------------ #
    def _build_left_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)

        gb_tool = QGroupBox("功能")
        v = QVBoxLayout(gb_tool)
        self.rb_tools = []
        for i, (name, tip) in enumerate([
            ("① 物体消除", "涂抹或框选要擦掉的物体"),
            ("② 水印消除", "自动检测或手动框选水印"),
            ("③ 文字消除", "OCR 识别后批量擦除文字"),
            ("④ 文字修改", "识别文字并替换为新内容，保持原风格"),
            ("⑤ Logo 修改", "去除旧 Logo 并贴合新 Logo"),
        ]):
            rb = QRadioButton(name)
            rb.setToolTip(tip)
            rb.toggled.connect(lambda ck, k=i: ck and self._switch_tool(k))
            v.addWidget(rb)
            self.rb_tools.append(rb)
        # 阻塞信号，避免 setChecked 在 self.stack 创建前触发 _switch_tool
        self.rb_tools[0].blockSignals(True)
        self.rb_tools[0].setChecked(True)
        self.rb_tools[0].blockSignals(False)
        lay.addWidget(gb_tool)

        gb_sel = QGroupBox("选区工具")
        v2 = QVBoxLayout(gb_sel)
        self.rb_brush = QRadioButton("画笔涂抹 (B)")
        self.rb_eraser = QRadioButton("橡皮擦除 (E)")
        self.rb_rect = QRadioButton("矩形框选 (R)")
        self.rb_quad = QRadioButton("四点拾取 (Q)")
        self.rb_brush.setChecked(True)
        for rb, m, key in ((self.rb_brush, Canvas.MODE_BRUSH, "B"),
                           (self.rb_eraser, Canvas.MODE_ERASE, "E"),
                           (self.rb_rect, Canvas.MODE_RECT, "R"),
                           (self.rb_quad, Canvas.MODE_QUAD, "Q")):
            rb.toggled.connect(lambda ck, mm=m: ck and self._set_canvas_mode(mm))
            v2.addWidget(rb)
            a = QAction(self); a.setShortcut(QKeySequence(key))
            a.triggered.connect(lambda _, r=rb: r.setChecked(True))
            self.addAction(a)

        hb = QHBoxLayout()
        hb.addWidget(QLabel("笔刷"))
        self.sl_brush = QSlider(Qt.Horizontal)
        self.sl_brush.setRange(2, 300); self.sl_brush.setValue(24)
        self.sl_brush.valueChanged.connect(
            lambda v_: setattr(self.canvas, "brush", v_))
        self.lb_brush = QLabel("24")
        self.sl_brush.valueChanged.connect(lambda v_: self.lb_brush.setText(str(v_)))
        hb.addWidget(self.sl_brush); hb.addWidget(self.lb_brush)
        v2.addLayout(hb)

        btn_clear = QPushButton("清除选区 (Del)")
        btn_clear.clicked.connect(self.canvas.clear_mask)
        a = QAction(self); a.setShortcut(QKeySequence(Qt.Key_Delete))
        a.triggered.connect(self.canvas.clear_mask); self.addAction(a)
        v2.addWidget(btn_clear)
        lay.addWidget(gb_sel)

        tips = QLabel(
            "<small>滚轮缩放 · 中键拖动平移<br>"
            "Shift+滚轮 调笔刷大小<br>"
            "Tab 对比原图</small>")
        tips.setStyleSheet("color:#777;")
        lay.addWidget(tips)
        lay.addStretch(1)
        return w

    # ------------------------------------------------------------------ #
    def _build_right_panel(self) -> QWidget:
        self.stack = QStackedWidget()
        self.stack.addWidget(self._panel_object())
        self.stack.addWidget(self._panel_watermark())
        self.stack.addWidget(self._panel_text_erase())
        self.stack.addWidget(self._panel_text_edit())
        self.stack.addWidget(self._panel_logo())

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.stack)
        scroll.setMinimumWidth(300)
        return scroll

    # ---------- 通用修复参数控件 ---------- #
    def _inpaint_param_box(self, prefix: str) -> QGroupBox:
        gb = QGroupBox("修复参数")
        f = QFormLayout(gb)

        cb = QComboBox()
        cb.addItem("自动（推荐）", "auto")
        cb.addItem("LaMa 纹理续写", "lama")
        cb.addItem("SD 生成式（需 GPU）", "sd")
        cb.addItem("OpenCV 传统（最快）", "opencv")
        setattr(self, f"{prefix}_method", cb)
        f.addRow("算法：", cb)

        sp_d = QSpinBox(); sp_d.setRange(0, 40); sp_d.setValue(4)
        sp_d.setToolTip("向外扩张选区，吃掉物体的抗锯齿边缘，避免残留轮廓")
        setattr(self, f"{prefix}_dilate", sp_d)
        f.addRow("选区膨胀：", sp_d)

        sp_f = QSpinBox(); sp_f.setRange(0, 40); sp_f.setValue(4)
        sp_f.setToolTip("回贴时的边缘羽化半径，越大过渡越柔和")
        setattr(self, f"{prefix}_feather", sp_f)
        f.addRow("边缘羽化：", sp_f)

        sp_c = QDoubleSpinBox(); sp_c.setRange(0.1, 3.0); sp_c.setSingleStep(0.1)
        sp_c.setValue(0.7)
        sp_c.setToolTip("送入模型的上下文范围，越大越慢但更懂周围纹理")
        setattr(self, f"{prefix}_ctx", sp_c)
        f.addRow("上下文比例：", sp_c)

        ck = QCheckBox("匹配噪点颗粒"); ck.setChecked(True)
        setattr(self, f"{prefix}_grain", ck)
        f.addRow("", ck)
        return gb

    def _collect_opt(self, prefix: str) -> InpaintOptions:
        return InpaintOptions(
            method=getattr(self, f"{prefix}_method").currentData(),
            dilate=getattr(self, f"{prefix}_dilate").value(),
            feather=getattr(self, f"{prefix}_feather").value(),
            context_ratio=getattr(self, f"{prefix}_ctx").value(),
            match_grain=getattr(self, f"{prefix}_grain").isChecked(),
        )

    # ---------- ① 物体消除 ---------- #
    def _panel_object(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        lay.addWidget(QLabel("<b>① 物体消除</b><br>"
                             "<small>用画笔涂抹或矩形框选要擦掉的物体，"
                             "然后点击下方按钮。</small>"))
        lay.addWidget(self._inpaint_param_box("obj"))

        b = QPushButton("擦除选区内的物体")
        b.setStyleSheet("font-weight:bold; padding:8px;")
        b.clicked.connect(lambda: self.run_object_erase("obj"))
        lay.addWidget(b)
        lay.addStretch(1)
        return w

    # ---------- ② 水印消除 ---------- #
    def _panel_watermark(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        lay.addWidget(QLabel("<b>② 水印消除</b><br>"
                             "<small>自动检测基于「重复文本 / 版权符号 / "
                             "半透明笔画 / 周期平铺」四条线索投票，"
                             "结果需人工确认。</small>"))

        b1 = QPushButton("自动检测水印")
        b1.clicked.connect(self.run_watermark_detect)
        lay.addWidget(b1)

        self.lst_wm = QListWidget()
        self.lst_wm.setSelectionMode(QAbstractItemView.MultiSelection)
        self.lst_wm.setMaximumHeight(180)
        self.lst_wm.itemSelectionChanged.connect(self._on_wm_select)
        lay.addWidget(self.lst_wm)

        self.chk_wm_tight = QCheckBox("收紧到笔画级（保留更多背景）")
        self.chk_wm_tight.setChecked(True)
        lay.addWidget(self.chk_wm_tight)

        b2 = QPushButton("将选中候选加入选区")
        b2.clicked.connect(self.wm_add_to_mask)
        lay.addWidget(b2)

        lay.addWidget(self._inpaint_param_box("wm"))

        b3 = QPushButton("擦除选区内的水印")
        b3.setStyleSheet("font-weight:bold; padding:8px;")
        b3.clicked.connect(lambda: self.run_object_erase("wm"))
        lay.addWidget(b3)
        lay.addStretch(1)
        return w

    # ---------- ③ 文字消除 ---------- #
    def _panel_text_erase(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        lay.addWidget(QLabel("<b>③ 文字消除</b><br>"
                             "<small>OCR 识别后勾选要擦的行；"
                             "也可以直接用画笔涂抹后点最下方按钮。</small>"))

        b1 = QPushButton("识别图中文字 (OCR)")
        b1.clicked.connect(self.run_ocr)
        lay.addWidget(b1)

        self.lst_text = QListWidget()
        self.lst_text.setSelectionMode(QAbstractItemView.MultiSelection)
        self.lst_text.setMaximumHeight(220)
        self.lst_text.itemSelectionChanged.connect(self._on_text_select)
        lay.addWidget(self.lst_text)

        hb = QHBoxLayout()
        b_all = QPushButton("全选"); b_all.clicked.connect(self.lst_text.selectAll)
        b_non = QPushButton("全不选"); b_non.clicked.connect(self.lst_text.clearSelection)
        hb.addWidget(b_all); hb.addWidget(b_non)
        lay.addLayout(hb)

        self.chk_txt_tight = QCheckBox("笔画级擦除（推荐）")
        self.chk_txt_tight.setChecked(True)
        lay.addWidget(self.chk_txt_tight)

        lay.addWidget(self._inpaint_param_box("txt"))

        b2 = QPushButton("擦除选中的文字")
        b2.setStyleSheet("font-weight:bold; padding:8px;")
        b2.clicked.connect(self.run_text_erase)
        lay.addWidget(b2)

        b3 = QPushButton("擦除画笔选区（不依赖 OCR）")
        b3.clicked.connect(lambda: self.run_object_erase("txt"))
        lay.addWidget(b3)
        lay.addStretch(1)
        return w

    # ---------- ④ 文字修改 ---------- #
    def _panel_text_edit(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        lay.addWidget(QLabel("<b>④ 文字修改</b><br>"
                             "<small>识别文字 → 在列表或画布上点选某一行 → "
                             "输入新文字。程序会自动继承颜色、字号、"
                             "倾斜、透视与质感。</small>"))

        b1 = QPushButton("识别图中文字 (OCR)")
        b1.clicked.connect(self.run_ocr)
        lay.addWidget(b1)

        self.lst_edit = QListWidget()
        self.lst_edit.setMaximumHeight(240)
        self.lst_edit.itemSelectionChanged.connect(self._on_edit_select)
        self.lst_edit.itemDoubleClicked.connect(lambda _: self.run_text_replace())
        lay.addWidget(self.lst_edit)

        self.lb_style = QLabel("<small>（选中一行后显示风格分析）</small>")
        self.lb_style.setWordWrap(True)
        self.lb_style.setStyleSheet("color:#666;")
        lay.addWidget(self.lb_style)

        b2 = QPushButton("修改选中文字…")
        b2.setStyleSheet("font-weight:bold; padding:8px;")
        b2.clicked.connect(self.run_text_replace)
        lay.addWidget(b2)

        b3 = QPushButton("在手动框选区域内添加文字…")
        b3.setToolTip("先用「四点拾取」在画布上点 4 个角，再点此按钮")
        b3.clicked.connect(self.run_text_on_quad)
        lay.addWidget(b3)
        lay.addStretch(1)
        return w

    # ---------- ⑤ Logo 修改 ---------- #
    def _panel_logo(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)
        lay.addWidget(QLabel("<b>⑤ Logo 修改</b><br>"
                             "<small>步骤：涂抹旧 Logo → 载入新 Logo → "
                             "用「四点拾取」在画布上按 <b>左上→右上→右下→左下</b> "
                             "顺序点 4 个角 → 一键替换。</small>"))

        b1 = QPushButton("载入新 Logo 图片…")
        b1.clicked.connect(self.load_logo)
        lay.addWidget(b1)
        self.lb_logo = QLabel("未载入")
        self.lb_logo.setStyleSheet("color:#666;")
        lay.addWidget(self.lb_logo)

        b2 = QPushButton("用当前选区的外接矩形作为放置区")
        b2.clicked.connect(self.logo_quad_from_mask)
        lay.addWidget(b2)
        self.lb_quad = QLabel("放置区：未指定")
        self.lb_quad.setStyleSheet("color:#666;")
        lay.addWidget(self.lb_quad)

        gb = QGroupBox("贴合参数")
        f = QFormLayout(gb)
        self.cb_blend = QComboBox()
        self.cb_blend.addItem("常规覆盖", "normal")
        self.cb_blend.addItem("正片叠底（印在布料/纸上）", "multiply")
        self.cb_blend.addItem("滤色（玻璃/灯箱）", "screen")
        f.addRow("混合模式：", self.cb_blend)

        self.sp_op = QDoubleSpinBox(); self.sp_op.setRange(0.05, 1.0)
        self.sp_op.setSingleStep(0.05); self.sp_op.setValue(1.0)
        f.addRow("不透明度：", self.sp_op)

        self.sp_shade = QDoubleSpinBox(); self.sp_shade.setRange(0.0, 1.0)
        self.sp_shade.setSingleStep(0.05); self.sp_shade.setValue(0.75)
        self.sp_shade.setToolTip("继承载体表面的明暗渐变强度——这是「贴得像」的关键")
        f.addRow("光照继承：", self.sp_shade)

        self.sp_illum = QDoubleSpinBox(); self.sp_illum.setRange(0.0, 1.0)
        self.sp_illum.setSingleStep(0.05); self.sp_illum.setValue(0.7)
        f.addRow("色调匹配：", self.sp_illum)

        self.chk_tex = QCheckBox("匹配锐度与噪点"); self.chk_tex.setChecked(True)
        f.addRow("", self.chk_tex)
        self.chk_ar = QCheckBox("保持 Logo 宽高比"); self.chk_ar.setChecked(True)
        f.addRow("", self.chk_ar)
        lay.addWidget(gb)

        lay.addWidget(self._inpaint_param_box("logo"))

        b3 = QPushButton("① 仅去除旧 Logo（用当前选区）")
        b3.clicked.connect(lambda: self.run_object_erase("logo"))
        lay.addWidget(b3)

        b4 = QPushButton("② 仅贴合新 Logo")
        b4.clicked.connect(self.run_logo_place)
        lay.addWidget(b4)

        b5 = QPushButton("一键替换（先擦后贴）")
        b5.setStyleSheet("font-weight:bold; padding:8px;")
        b5.clicked.connect(self.run_logo_replace)
        lay.addWidget(b5)
        lay.addStretch(1)
        return w

    # ------------------------------------------------------------------ #
    def _build_statusbar(self):
        sb = QStatusBar()
        self.setStatusBar(sb)
        self.lb_status = QLabel("就绪")
        self.lb_pos = QLabel("")
        self.pb = QProgressBar()
        self.pb.setMaximumWidth(220); self.pb.setVisible(False)
        sb.addWidget(self.lb_status, 1)
        sb.addPermanentWidget(self.lb_pos)
        sb.addPermanentWidget(self.pb)

    # ================================================================== #
    # 通用逻辑
    # ================================================================== #
    def _update_status(self, text: str):
        self.lb_status.setText(text)

    def _on_cursor(self, x, y):
        self.lb_pos.setText(f"({x}, {y})")

    def _set_canvas_mode(self, mode: str):
        self.canvas.mode = mode
        self.canvas.clear_quad_picking()

    def _switch_tool(self, idx: int):
        self.stack.setCurrentIndex(idx)
        if idx == 4:
            self.rb_quad.setChecked(False)

    def _on_compare(self, state):
        self.canvas.show_original = bool(state)
        self.canvas.update()

    # ---------- 当前页图像存取 ---------- #
    @property
    def cur_image(self) -> Optional[np.ndarray]:
        return self.canvas.image

    def _set_cur_image(self, img: np.ndarray, push_undo: bool = True):
        if push_undo and self.canvas.image is not None:
            self._undo.append(self.canvas.image.copy())
            if len(self._undo) > 20:
                self._undo.pop(0)
            self._redo.clear()
        self.canvas.set_image(img, keep_view=True, reset_mask=False)
        if self.pages:
            self.pages[self.page_idx].image = img
            self.edited_pages.add(self.page_idx)

    def undo(self):
        if not self._undo:
            return
        self._redo.append(self.canvas.image.copy())
        img = self._undo.pop()
        self.canvas.set_image(img, keep_view=True, reset_mask=False)
        if self.pages:
            self.pages[self.page_idx].image = img
            self.edited_pages.add(self.page_idx)
        self._update_status("已撤销")

    def redo(self):
        if not self._redo:
            return
        self._undo.append(self.canvas.image.copy())
        img = self._redo.pop()
        self.canvas.set_image(img, keep_view=True, reset_mask=False)
        if self.pages:
            self.pages[self.page_idx].image = img
            self.edited_pages.add(self.page_idx)
        self._update_status("已重做")

    # ---------- 后台任务框架 ---------- #
    def _run(self, fn, on_done, title: str = "处理中…"):
        """在后台线程执行 fn，完成后在主线程调用 on_done(result)。"""
        if self._busy:
            QMessageBox.information(self, "请稍候", "上一个任务还在执行中。")
            return
        self._busy = True
        self.pb.setVisible(True)
        self.pb.setRange(0, 100)
        self.pb.setValue(0)
        self._update_status(title)
        self.setEnabled(True)

        w = Worker(fn)
        self._worker = w

        def _prog(p, t):
            self.pb.setValue(int(np.clip(p, 0, 1) * 100))
            if t:
                self._update_status(t)

        def _done(res):
            self._busy = False
            self.pb.setVisible(False)
            try:
                on_done(res)
            except Exception:
                LOG.error(traceback.format_exc())
            self._update_status("完成 · " + self.service.describe())

        def _fail(msg):
            self._busy = False
            self.pb.setVisible(False)
            LOG.error(msg)
            QMessageBox.critical(self, "出错了", msg[-2500:])
            self._update_status("出错")

        w.sig_progress.connect(_prog)
        w.sig_done.connect(_done)
        w.sig_failed.connect(_fail)
        # 把 service 的进度回调接到本线程信号
        self.service._progress = lambda p, t="": w.sig_progress.emit(p, t)
        w.start()

    # ================================================================== #
    # 文件操作
    # ================================================================== #
    def open_file(self):
        filt = "支持的文件 (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.pdf);;所有文件 (*)"
        path, _ = QFileDialog.getOpenFileName(self, "打开图片或 PDF", "", filt)
        if not path:
            return
        self.load_path(path)

    def load_path(self, path: str):
        ext = os.path.splitext(path)[1].lower()
        self.src_path = path
        self._undo.clear(); self._redo.clear()
        self.ocr_boxes = []; self.wm_cands = []
        self.lst_text.clear(); self.lst_edit.clear(); self.lst_wm.clear()
        self.canvas.set_boxes([])

        if ext == ".pdf":
            if not HAS_FITZ:
                QMessageBox.warning(self, "缺少依赖",
                                    "处理 PDF 需要 PyMuPDF：\npip install PyMuPDF")
                return
            self.is_pdf = True
            info = probe_pdf(path)
            self._update_status(f"正在光栅化 PDF…（{info.splitlines()[0]}）")
            QApplication.processEvents()
            self.pages = pdf_to_pages(path, dpi=300, adaptive=True)
        else:
            self.is_pdf = False
            img = imread_rgb(path)
            h, w = img.shape[:2]
            self.pages = [PdfPage(0, img, w * 72 / 300, h * 72 / 300, 300.0)]

        self.page_idx = 0
        self.edited_pages.clear()
        self.cb_page.blockSignals(True)
        self.cb_page.clear()
        for i, p in enumerate(self.pages):
            self.cb_page.addItem(f"{i + 1}/{len(self.pages)}"
                                 f"  {p.size_px[0]}×{p.size_px[1]}")
        self.cb_page.setCurrentIndex(0)
        self.cb_page.blockSignals(False)

        self._load_page(0)
        self._update_status(
            f"已打开 {os.path.basename(path)} · {len(self.pages)} 页 · "
            + self.service.describe())

    def _load_page(self, idx: int):
        if not self.pages:
            return
        idx = int(np.clip(idx, 0, len(self.pages) - 1))
        self.page_idx = idx
        p = self.pages[idx]
        self.canvas.set_image(p.image, keep_view=False, reset_mask=True)
        self.canvas.set_original(p.image.copy())
        self.canvas.set_boxes([])
        self.ocr_boxes = []
        self.lst_text.clear(); self.lst_edit.clear(); self.lst_wm.clear()
        self._undo.clear(); self._redo.clear()

    def _on_page_change(self, idx):
        if idx >= 0:
            self._load_page(idx)

    def save_file(self):
        if self.cur_image is None:
            return
        base = os.path.splitext(os.path.basename(self.src_path or "output"))[0]
        suffix = f"_p{self.page_idx + 1}" if len(self.pages) > 1 else ""
        default = os.path.join(os.path.dirname(self.src_path or "."),
                               f"{base}{suffix}_edited.png")
        path, _ = QFileDialog.getSaveFileName(
            self, "保存图片", default,
            "PNG 无损 (*.png);;JPEG (*.jpg);;WEBP 无损 (*.webp);;TIFF (*.tif)")
        if not path:
            return
        imwrite_rgb(path, self.cur_image)
        self._update_status(f"已保存：{path}")
        QMessageBox.information(self, "已保存", path)

    def export_pdf(self):
        if not self.pages:
            return
        base = os.path.splitext(os.path.basename(self.src_path or "output"))[0]
        default = os.path.join(os.path.dirname(self.src_path or "."),
                               f"{base}_edited.pdf")
        path, _ = QFileDialog.getSaveFileName(self, "导出 PDF", default,
                                              "PDF 文件 (*.pdf)")
        if not path:
            return
        ret = QMessageBox.question(
            self, "画质选择",
            "使用无损 PNG 内嵌吗？\n\n"
            "「是」= PNG 无损，画质最好、体积大\n"
            "「否」= JPEG q95，体积可小 5~10 倍，肉眼几乎无差",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        lossless = (ret == QMessageBox.Yes)

        def job():
            try:
                if (self.src_path and self.src_path.lower().endswith(".pdf")
                        and self.edited_pages):
                    return pages_to_pdf_preserved(
                        self.src_path, self.pages, sorted(self.edited_pages),
                        path, lossless=lossless)
            except Exception as e:
                LOG.warning("保留未改页导出失败，回退全量光栅化：%s", e)
            return pages_to_pdf(self.pages, path, lossless=lossless)

        self._run(job, lambda r: QMessageBox.information(self, "已导出", str(r)),
                  "正在导出 PDF…")

    # ================================================================== #
    # ① 物体 / 通用消除
    # ================================================================== #
    def run_object_erase(self, prefix: str = "obj"):
        img = self.cur_image
        if img is None:
            return
        mask = self.canvas.mask
        if mask is None or mask.max() == 0:
            QMessageBox.information(self, "没有选区",
                                    "请先用画笔涂抹或矩形框选要处理的区域。")
            return
        opt = self._collect_opt(prefix)
        m = mask.copy()

        if prefix == "wm" and self.chk_wm_tight.isChecked():
            m = refine_mask_for_watermark(img, m)

        def job():
            return self.service.inpaint(img, m, opt)

        def done(res):
            self._set_cur_image(res)
            self.canvas.clear_mask()

        self._run(job, done, "正在修复…")

    # ================================================================== #
    # ② 水印
    # ================================================================== #
    def run_watermark_detect(self):
        img = self.cur_image
        if img is None:
            return

        def job():
            boxes = self.ocr.detect(img) if self.ocr.load() else []
            return auto_detect(img, boxes)

        def done(res):
            mask, cands, diag = res
            self.wm_cands = cands
            self.lst_wm.clear()
            for c in cands:
                x0, y0, x1, y1 = c.bbox
                txt = (c.text[:16] + "…") if len(c.text) > 16 else c.text
                it = QListWidgetItem(
                    f"[{c.score:.2f}] {txt or '（无文字）'}  "
                    f"{x1 - x0}×{y1 - y0} @({x0},{y0})  ← {c.reason}")
                self.lst_wm.addItem(it)
            self.canvas.set_boxes([c.quad for c in cands],
                                  [f"{c.score:.2f}" for c in cands])
            self._update_status("水印检测：" + diag)
            if not cands:
                QMessageBox.information(
                    self, "未检测到水印",
                    "自动检测没有找到高置信度的水印。\n"
                    "请改用画笔或矩形框手动选择水印区域。")

        self._run(job, done, "正在检测水印（首次会加载 OCR 模型）…")

    def _on_wm_select(self):
        sel = {i.row() for i in self.lst_wm.selectedIndexes()}
        self.canvas.box_selected = sel
        self.canvas.update()

    def wm_add_to_mask(self):
        if not self.wm_cands:
            return
        rows = [i.row() for i in self.lst_wm.selectedIndexes()]
        if not rows:
            QMessageBox.information(self, "未选择", "请先在列表中选中候选项。")
            return
        h, w = self.cur_image.shape[:2]
        m = np.zeros((h, w), np.uint8)
        for r in rows:
            m = cv2.bitwise_or(m, quad_to_mask((h, w), self.wm_cands[r].quad))
        self.canvas.add_to_mask(m)
        self._update_status(f"已把 {len(rows)} 个候选加入选区")

    # ================================================================== #
    # ③④ OCR 与文字
    # ================================================================== #
    def run_ocr(self):
        img = self.cur_image
        if img is None:
            return

        def job():
            return self.ocr.detect(img)

        def done(boxes):
            self.ocr_boxes = boxes
            self.lst_text.clear(); self.lst_edit.clear()
            for i, b in enumerate(boxes):
                x0, y0, x1, y1 = b.bbox
                label = f"{i + 1:02d}  {b.text}   [{b.score:.2f}] @({x0},{y0})"
                self.lst_text.addItem(QListWidgetItem(label))
                self.lst_edit.addItem(QListWidgetItem(label))
            self.canvas.set_boxes([b.quad for b in boxes],
                                  [str(i + 1) for i in range(len(boxes))])
            if not boxes:
                QMessageBox.information(
                    self, "未识别到文字",
                    f"OCR 后端：{OcrEngine.probe()}\n\n"
                    "若未安装 OCR，请执行：\n"
                    "  pip install paddlepaddle paddleocr\n或\n  pip install easyocr\n\n"
                    "也可以直接用画笔涂抹文字区域后手动擦除。")
            self._update_status(f"OCR 完成，共 {len(boxes)} 行文字")

        self._run(job, done, "正在识别文字（首次会下载 OCR 模型）…")

    def _on_text_select(self):
        self.canvas.box_selected = {i.row() for i in self.lst_text.selectedIndexes()}
        self.canvas.update()

    def _on_edit_select(self):
        rows = [i.row() for i in self.lst_edit.selectedIndexes()]
        self.canvas.box_selected = set(rows)
        self.canvas.update()
        if rows and self.cur_image is not None:
            b = self.ocr_boxes[rows[0]]
            st = estimate_style(self.cur_image, b.quad, sample_text=b.text)
            self.lb_style.setText(f"<small>{st.describe()}</small>")

    def _on_box_clicked(self, idx: int):
        """点击画布上的框 → 同步选中列表项。"""
        cur = self.stack.currentIndex()
        lst = {2: self.lst_text, 3: self.lst_edit, 1: self.lst_wm}.get(cur)
        if lst and 0 <= idx < lst.count():
            lst.setCurrentRow(idx)

    def run_text_erase(self):
        img = self.cur_image
        if img is None or not self.ocr_boxes:
            QMessageBox.information(self, "请先识别", "请先点击「识别图中文字」。")
            return
        rows = [i.row() for i in self.lst_text.selectedIndexes()]
        if not rows:
            QMessageBox.information(self, "未选择", "请在列表中选中要擦除的文字行。")
            return
        quads = [self.ocr_boxes[r].quad for r in rows]
        tight = self.chk_txt_tight.isChecked()
        opt = self._collect_opt("txt")

        def job():
            return erase_text(img, quads, self.service, tight=tight, opt=opt)

        def done(res):
            self._set_cur_image(res)
            self.canvas.set_boxes([])
            self.ocr_boxes = []
            self.lst_text.clear(); self.lst_edit.clear()

        self._run(job, done, f"正在擦除 {len(rows)} 行文字…")

    def run_text_replace(self):
        img = self.cur_image
        if img is None or not self.ocr_boxes:
            QMessageBox.information(self, "请先识别", "请先点击「识别图中文字」。")
            return
        rows = [i.row() for i in self.lst_edit.selectedIndexes()]
        if not rows:
            QMessageBox.information(self, "未选择", "请在列表中选中一行文字。")
            return
        box = self.ocr_boxes[rows[0]]
        style = estimate_style(img, box.quad, sample_text=box.text)

        dlg = TextEditDialog(box, style, self)
        if dlg.exec_() != QDialog.Accepted:
            return
        v = dlg.values()
        if not v["new_text"]:
            QMessageBox.information(self, "内容为空", "请输入新文字。")
            return

        def job():
            out, _ = replace_text(img, box, v["new_text"], self.service,
                                  fit_mode=v["fit_mode"],
                                  tight_erase=v["tight"],
                                  color_override=v["color"],
                                  font_override=v["font"],
                                  size_scale=v["size_scale"],
                                  style=style)
            return out

        def done(res):
            self._set_cur_image(res)
            self.canvas.set_boxes([])
            self.ocr_boxes = []
            self.lst_text.clear(); self.lst_edit.clear()

        self._run(job, done, "正在替换文字…")

    def run_text_on_quad(self):
        """在用户手动拾取的四边形内添加/替换文字（无需 OCR）。"""
        img = self.cur_image
        if img is None:
            return
        if self.logo_quad is None:
            QMessageBox.information(
                self, "请先拾取区域",
                "请先在左侧选择「四点拾取」，在画布上按\n"
                "左上 → 右上 → 右下 → 左下 的顺序点 4 个角。")
            return
        quad = self.logo_quad
        pseudo = TextBox(order_quad(quad), "", 1.0)
        style = estimate_style(img, quad, sample_text="示例Aa")
        dlg = TextEditDialog(pseudo, style, self)
        if dlg.exec_() != QDialog.Accepted:
            return
        v = dlg.values()
        if not v["new_text"]:
            return

        def job():
            base = erase_text(img, [quad], self.service, tight=v["tight"]) \
                if v["tight"] else img
            out, _ = render_new_text(base, v["new_text"], style,
                                     fit_mode=v["fit_mode"],
                                     color_override=v["color"],
                                     font_override=v["font"],
                                     size_scale=v["size_scale"])
            return out

        self._run(job, lambda r: self._set_cur_image(r), "正在生成文字…")

    # ================================================================== #
    # ⑤ Logo
    # ================================================================== #
    def load_logo(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择新 Logo 图片", "",
            "图片 (*.png *.jpg *.jpeg *.webp *.bmp);;所有文件 (*)")
        if not path:
            return
        self.logo_rgba = imread_rgba(path)
        h, w = self.logo_rgba.shape[:2]
        self.lb_logo.setText(f"已载入：{os.path.basename(path)}  {w}×{h}")
        self._update_status("Logo 已载入，请用「四点拾取」指定放置区域")
        self.rb_quad.setChecked(True)

    def _on_quad(self, quad):
        self.logo_quad = np.asarray(quad, np.float32)
        q = order_quad(self.logo_quad).astype(int)
        self.lb_quad.setText(f"放置区：{q.tolist()}")
        self.canvas.set_boxes([self.logo_quad], ["放置区"])
        self._update_status("已指定放置区域")

    def _on_rect(self, rect):
        """矩形框选 → 直接写入掩膜（并记为潜在的 Logo 放置区）。"""
        x0, y0, x1, y1 = rect
        if self.cur_image is None:
            return
        h, w = self.cur_image.shape[:2]
        m = np.zeros((h, w), np.uint8)
        cv2.rectangle(m, (max(0, x0), max(0, y0)),
                      (min(w - 1, x1), min(h - 1, y1)), 255, -1)
        self.canvas.add_to_mask(m)

    def logo_quad_from_mask(self):
        m = self.canvas.mask
        if m is None or m.max() == 0:
            QMessageBox.information(self, "没有选区", "请先涂抹或框选旧 Logo 区域。")
            return
        ys, xs = np.where(m > 0)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        self.logo_quad = quad_from_bbox(bbox)
        self.lb_quad.setText(f"放置区（来自选区）：{bbox}")
        self.canvas.set_boxes([self.logo_quad], ["放置区"])

    def _logo_opt(self) -> LogoOptions:
        return LogoOptions(
            blend_mode=self.cb_blend.currentData(),
            opacity=float(self.sp_op.value()),
            shading_strength=float(self.sp_shade.value()),
            illum_strength=float(self.sp_illum.value()),
            match_texture=self.chk_tex.isChecked(),
            keep_aspect=self.chk_ar.isChecked(),
        )

    def run_logo_place(self):
        img = self.cur_image
        if img is None:
            return
        if self.logo_rgba is None or self.logo_quad is None:
            QMessageBox.information(self, "信息不全",
                                    "请先载入 Logo 图片并指定放置区域。")
            return
        logo, quad, opt = self.logo_rgba, self.logo_quad, self._logo_opt()

        def job():
            out, _ = place_logo(img, logo, quad, opt)
            return out

        self._run(job, lambda r: self._set_cur_image(r), "正在贴合 Logo…")

    def run_logo_replace(self):
        img = self.cur_image
        if img is None:
            return
        mask = self.canvas.mask
        if (mask is None or mask.max() == 0) and self.logo_rgba is None:
            QMessageBox.information(self, "信息不全",
                                    "请至少涂抹旧 Logo 区域，或载入新 Logo。")
            return
        m = mask.copy() if mask is not None and mask.max() > 0 else None
        if self.logo_quad is None and m is not None:
            self.logo_quad_from_mask()
        logo, quad = self.logo_rgba, self.logo_quad
        lopt, iopt = self._logo_opt(), self._collect_opt("logo")

        def job():
            return replace_logo(img, m, logo, quad, self.service, lopt, iopt)

        def done(res):
            self._set_cur_image(res)
            self.canvas.clear_mask()

        self._run(job, done, "正在替换 Logo…")

    # ------------------------------------------------------------------ #
    def resizeEvent(self, ev):
        super().resizeEvent(ev)

    def closeEvent(self, ev):
        if self._busy:
            r = QMessageBox.question(self, "任务进行中",
                                     "还有任务在执行，确定退出吗？",
                                     QMessageBox.Yes | QMessageBox.No)
            if r != QMessageBox.Yes:
                ev.ignore()
                return
        ev.accept()


# --------------------------------------------------------------------------- #
def create_app(argv=None):
    """创建 QApplication 与主窗口（供 main.py 调用）。"""
    app = QApplication(argv or sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei" if sys.platform == "win32" else "Sans", 9))
    win = MainWindow()
    return app, win
