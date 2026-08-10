# -*- coding: utf-8 -*-
# ============================================================================
# PyInstaller 打包配置（onedir 模式）
# ----------------------------------------------------------------------------
# 关键策略
#   1. 排除 torch / paddle / diffusers 等重型库
#      -> EXE 体积更小，且避开 Windows 上 c10.dll 的 VC++ 运行时依赖问题
#   2. 内置 LaMa-ONNX（onnxruntime）+ OpenCV + PyQt5 + PyMuPDF -> 开箱即用
#   3. onnxruntime / fitz / requests 为懒加载，必须显式加入 hiddenimports
#
# 用法:  pyinstaller build.spec
# 产物:  dist/smart_image_editor/ （一个文件夹，整体分发）
# ============================================================================

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        # 基础库（部分在业务里懒加载，必须显式声明才能打进 EXE）
        'cv2', 'numpy', 'PIL', 'PIL.Image', 'PIL.ImageDraw', 'PIL.ImageFont',
        'onnxruntime', 'fitz', 'requests', 'tqdm',
        'PyQt5', 'PyQt5.sip',
        'PyQt5.QtCore', 'PyQt5.QtGui', 'PyQt5.QtWidgets',
        # 业务包
        'core', 'core.inpaint', 'core.logo_edit', 'core.pdf_io',
        'core.text_edit', 'core.watermark',
        'models', 'models.downloader', 'models.lama',
        'models.ocr', 'models.sd_inpaint',
    ],
    excludes=[
        # 深度学习重型库：EXE 走 ONNX 路径即可，不打包它们
        'torch', 'torchvision', 'torchaudio',
        'paddle', 'paddleocr', 'paddlepaddle', 'paddlehub',
        'easyocr',
        'diffusers', 'transformers', 'accelerate', 'safetensors', 'xformers',
        'tensorflow', 'keras',
        'onnx',                 # 训练用 onnx，不是 onnxruntime
        'matplotlib', 'scipy', 'skimage', 'scikit-image',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='smart_image_editor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,            # 窗口程序，不弹控制台黑框
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name='smart_image_editor',
)
