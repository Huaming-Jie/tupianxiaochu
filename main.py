# -*- coding: utf-8 -*-
"""
main.py —— 程序入口
================================================================================
职责很轻，但有两件事必须在这里、而且必须在最前面做：

1. **把项目根目录塞进 sys.path**
   这样 ``ui.py`` 里的 ``from core.inpaint import ...`` 在任何工作目录下
   （双击运行、IDE 运行、命令行 cd 到别处运行）都能找到模块。

2. **设置 HuggingFace 镜像端点**
   ``HF_ENDPOINT`` 环境变量只在 ``huggingface_hub`` **首次被导入时**读取。
   一旦 diffusers / paddleocr 等库先一步把它导进来，再设置就不生效了。
   所以这一步必须早于任何模型库的导入。

命令行用法::

    python main.py                     # 启动图形界面
    python main.py 图片.png            # 启动并直接打开文件
    python main.py --device cpu        # 强制使用 CPU
    python main.py --check             # 只做环境自检，不开界面
"""

from __future__ import annotations

import argparse
import os
import sys

# --- ① 路径 ---------------------------------------------------------------- #
# 冻结（PyInstaller）后 __file__ 不稳定，改用 sys.executable 所在目录。
if getattr(sys, "frozen", False):
    ROOT = os.path.dirname(os.path.abspath(sys.executable))
else:
    ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# --- ② 镜像（必须早于任何 HF 相关导入） -------------------------------------- #
from models.downloader import set_hf_mirror   # noqa: E402

set_hf_mirror()

from utils import LOG, MODEL_DIR, ensure_dir   # noqa: E402


# --------------------------------------------------------------------------- #
def environment_report() -> str:
    """生成一份环境自检报告，帮用户快速定位"为什么某功能不可用"。"""
    lines = ["=" * 66, "环境自检", "=" * 66,
             f"Python      : {sys.version.split()[0]}  ({sys.executable})",
             f"模型目录    : {MODEL_DIR}"]

    def chk(name, fn):
        try:
            lines.append(f"{name:<12}: {fn()}")
        except Exception as e:
            lines.append(f"{name:<12}: ✗ {e}")

    chk("NumPy", lambda: __import__("numpy").__version__)
    chk("OpenCV", lambda: __import__("cv2").__version__)
    chk("Pillow", lambda: __import__("PIL").__version__)

    def _qt():
        from PyQt5.QtCore import QT_VERSION_STR
        return f"✓ Qt {QT_VERSION_STR}"
    chk("PyQt5", _qt)

    def _fitz():
        import fitz
        return f"✓ {fitz.__doc__.strip().splitlines()[0]}"
    chk("PyMuPDF", _fitz)

    def _torch():
        import torch
        dev = "CUDA " + torch.cuda.get_device_name(0) \
            if torch.cuda.is_available() else "仅 CPU"
        return f"✓ {torch.__version__}  ({dev})"
    chk("PyTorch", _torch)

    def _ort():
        import onnxruntime as ort
        return f"✓ {ort.__version__}  providers={ort.get_available_providers()}"
    chk("onnxruntime", _ort)

    from models.ocr import OcrEngine
    lines.append(f"{'OCR 引擎':<12}: {OcrEngine.probe()}")

    from models.sd_inpaint import SDInpainter
    ok, msg = SDInpainter.probe()
    lines.append(f"{'SD 修复':<12}: {'✓' if ok else '✗'} {msg}")

    from models.downloader import MODEL_REGISTRY, model_exists
    for k, spec in MODEL_REGISTRY.items():
        state = "✓ 已就绪" if model_exists(k) else f"× 未下载（{spec.size_hint}，首次使用时自动下载）"
        lines.append(f"{'  ' + k:<12}: {state}")

    from utils import list_available_fonts
    fonts = list_available_fonts(True)
    lines.append(f"{'中文字体':<12}: 找到 {len(fonts)} 个"
                 + (f"，首选 {os.path.basename(fonts[0])}" if fonts else "（文字修改功能不可用）"))
    lines.append("=" * 66)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="智能图像编辑器 —— 物体/水印/文字消除、文字修改、Logo 替换")
    parser.add_argument("file", nargs="?", help="启动时直接打开的图片或 PDF")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda"], help="推理设备")
    parser.add_argument("--check", action="store_true", help="只做环境自检")
    parser.add_argument("--predownload", action="store_true",
                        help="预先下载 LaMa 权重后退出")
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[WARN] 忽略外部注入的参数: {unknown}（通常是 PaddlePaddle/OCR 库在 Windows 上的 argv 污染）")

    ensure_dir(MODEL_DIR)
    print(environment_report())

    if args.check:
        return 0

    if args.predownload:
        from models.downloader import get_model_path
        p = get_model_path("lama_ts",
                           progress=lambda c, t, m: print(f"\r{m}", end=""))
        print()
        print("下载结果：", p or "失败，请参考上方提示手动下载")
        return 0 if p else 1

    # --- 启动 GUI --- #
    try:
        from ui import create_app
    except ImportError as e:
        print(f"\n[致命] 无法加载图形界面：{e}\n"
              f"请先安装依赖：pip install -r requirements.txt\n")
        return 1

    app, win = create_app(sys.argv)
    win.show()

    if args.file and os.path.isfile(args.file):
        win.load_path(args.file)

    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
