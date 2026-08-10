# 智能图像编辑器 · Smart Image Editor

一个基于 Python + PyQt5 的本地图像修复与编辑工具，聚焦五件事：

| # | 功能 | 说明 |
|---|------|------|
| ① | **物体消除** | 涂抹/框选任意物体，自动擦除并续写背景纹理 |
| ② | **水印消除** | 四条线索自动检测水印，也支持手动框选 |
| ③ | **文字消除** | OCR 定位后**笔画级**擦除，不留糊块 |
| ④ | **文字修改** | 识别原文字 → 输入新文字，自动保持颜色/字号/倾斜/透视/质感 |
| ⑤ | **Logo 修改** | 去除旧 Logo，导入新 Logo 并自动适配透视、光照、混合模式 |

另外：**原生支持 PDF**（按页自适应 DPI 光栅化 → 编辑 → 无损回写），
以及 PNG / JPG / JPEG / WEBP / BMP / TIFF 的读写。

---

## 零、安装方式（二选一）

**方式 A · 源码运行（推荐开发者 / 需要 OCR / SD 增强）**

```bash
git clone <你的仓库地址> && cd smart_image_editor
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch --index-url https://download.pytorch.org/whl/cpu   # 可选，提升质量
python main.py --check     # 环境自检
python main.py             # 启动
```

**方式 B · 下载 EXE（Windows 用户，零依赖）**

到 **Releases** 页下载 `smart_image_editor-<版本>-win64.zip`，解压即运行。
首次启动会自动联网下载模型权重（约 400 MB），也可先跑：

```bash
python download_models.py        # 提前拉取权重
```

> 发布的 EXE 已内置 LaMa-ONNX 引擎，**无需安装 PyTorch / VC++ 运行时**即可擦除；
> OCR 与 Stable Diffusion 增强仅源码运行方式可用。

---

## 一、快速开始

```bash
# 1) 建议使用虚拟环境
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 2) 安装必装依赖（国内加清华镜像会快很多）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3) 安装 PyTorch（决定修复质量，强烈推荐）
#    CPU 版：
pip install torch --index-url https://download.pytorch.org/whl/cpu
#    有 N 卡（CUDA 12.1）：
# pip install torch --index-url https://download.pytorch.org/whl/cu121

# 4) 环境自检 —— 一眼看清哪些功能可用
python main.py --check

# 5) 启动
python main.py
# 或直接打开某个文件
python main.py D:\photos\test.jpg
```

### 可选：安装 OCR（③④ 功能与水印文本线索依赖它）

```bash
# 方案 A：PaddleOCR，中文精度最好
pip install paddlepaddle paddleocr -i https://pypi.tuna.tsinghua.edu.cn/simple
# 方案 B：EasyOCR，安装更轻
pip install easyocr
```

### 可选：安装 Stable Diffusion 生成式修复（需 GPU ≥ 6GB 显存）

```bash
pip install diffusers transformers accelerate safetensors
```

---

## 二、模型权重

程序**首次需要时自动下载**，全部走国内镜像 `hf-mirror.com`，
并带断点续传与 MD5 校验。权重存放在 `./models_zoo/`（可用环境变量
`SIE_MODEL_DIR` 改到别处）。

| 模型 | 体积 | 用途 | 何时下载 |
|------|------|------|----------|
| `big-lama.pt` | ≈196 MB | 主力修复引擎 | 第一次执行任意擦除 |
| PaddleOCR 检测/识别 | ≈15 MB | 文字定位与识别 | 第一次点「识别文字」 |
| SD Inpainting | ≈2.5 GB | 生成式修复（可选） | 手动选择该算法时 |

**提前下载**：

```bash
python download_models.py            # 推荐：清晰进度、可只下某一项
# 或等价的内置命令：
python main.py --predownload
```

**完全离线部署**：把 `big-lama.pt` 手动放进 `./models_zoo/` 即可。
下载地址（任选其一）：

- `https://hf-mirror.com/Sanster/models/resolve/main/big-lama.pt`
- `https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt`

若权重始终不可用，程序**不会崩溃**，会自动降级为 OpenCV 传统修复
（质量明显下降，但功能完整可用）。

---

## 三、界面与操作

```
┌─────────────────────────────────────────────────────────────┐
│ 打开 保存 导出PDF │ 撤销 重做 │ 适应窗口 1:1 │ □对比原图 │ 页▾ │
├──────────┬──────────────────────────────────┬───────────────┤
│ 功能     │                                  │ 参数面板      │
│ ①物体    │             画  布               │（随功能切换） │
│ ②水印    │                                  │               │
│ ③文字    │   涂抹 / 框选 / 四点拾取         │               │
│ ④改字    │                                  │               │
│ ⑤Logo    │                                  │               │
├──────────┴──────────────────────────────────┴───────────────┤
│ 引擎状态 │ 光标坐标 │ 进度条                                │
└─────────────────────────────────────────────────────────────┘
```

### 快捷键

| 键 | 作用 |
|----|------|
| `B` / `E` | 画笔涂抹 / 橡皮擦除 |
| `R` / `Q` | 矩形框选 / 四点拾取 |
| `Del` | 清除选区 |
| `Tab` | 切换「对比原图」 |
| 滚轮 | 以光标为锚点缩放 |
| `Shift`+滚轮 | 调整笔刷大小 |
| 中键拖动 | 平移画布 |
| `Ctrl+Z / Ctrl+Y` | 撤销 / 重做 |
| `Ctrl+O / Ctrl+S` | 打开 / 保存 |
| `Ctrl+Shift+S` | 导出 PDF |

### 五个功能的典型操作流

**① 物体消除**
1. 用画笔把物体涂红（涂到略微超出物体边缘）
2. 右侧点「擦除选区内的物体」

**② 水印消除**
1. 点「自动检测水印」→ 画布出现黄框、列表出现候选（带置信度与命中原因）
2. 在列表里勾选真正的水印 → 点「将选中候选加入选区」
3. 建议勾选「收紧到笔画级」→ 点「擦除选区内的水印」
   - 检测不到时直接用画笔涂抹，效果一样

**③ 文字消除**
1. 点「识别图中文字」
2. 在列表勾选要擦的行（可全选）
3. 点「擦除选中的文字」

**④ 文字修改**
1. 点「识别图中文字」→ 选中一行（列表或直接点画布上的框）
2. 下方会显示风格分析：`颜色 #1A1A1A｜字高 32px｜常规｜倾斜 -2.3°｜字体 msyh.ttc`
3. 点「修改选中文字…」，在对话框里输入新文字
   - **排版方式**：`填满原框`（长度变化大时用）/ `保持原字号`（要求字号严格一致时用）
   - **文字颜色**：默认沿用从原图聚类出来的颜色
4. 确定

**⑤ Logo 修改**
1. 用画笔涂抹旧 Logo
2. 点「载入新 Logo 图片…」（PNG 带透明通道最佳；白底 JPG 会自动去白底）
3. 左侧切到「四点拾取」，在画布上按 **左上 → 右上 → 右下 → 左下** 点 4 个角
   - 想偷懒可以点「用当前选区的外接矩形作为放置区」
4. 调「光照继承」「混合模式」→ 点「一键替换（先擦后贴）」

---

## 四、代码结构

```
smart_image_editor/
├── main.py                 入口：路径注入、HF 镜像设置、环境自检、启动 GUI
├── ui.py                   PyQt5 界面：画布、工具面板、后台线程
├── utils.py                地基：色彩/掩膜/几何/ROI 回贴/字体/文本渲染
├── requirements.txt
├── README.md
├── models/                 「把模型跑起来」
│   ├── downloader.py       多源候选 + 断点续传 + MD5 校验
│   ├── lama.py             LaMa：TorchScript → ONNX → OpenCV 三级降级
│   ├── sd_inpaint.py       Stable Diffusion Inpainting（懒加载、可选）
│   └── ocr.py              PaddleOCR / EasyOCR 统一封装
├── core/                   「用模型解决问题」
│   ├── inpaint.py          ROI 调度、连通域拆分、噪点匹配、无损回贴
│   ├── watermark.py        四线索水印检测 + 笔画级掩膜收紧
│   ├── text_edit.py        文字擦除 + 七维风格估计 + 风格化重绘
│   ├── logo_edit.py        Logo 透视/光照/质感融合，三种混合模式
│   └── pdf_io.py           PDF 自适应 DPI 光栅化与无损回写
└── models_zoo/             模型权重（自动创建）
```

---

## 五、关键设计说明

### 5.1 为什么"其他元素绝对不受损"能得到保证

所有修复都遵循同一条流水线：

```
原图 ──┬─────────────────────────────────────────────┐
       │                                             │
       └─ 裁 ROI ─→ 模型修复 ─→ 质感匹配 ─→ 羽化 α ──┴─→ 合成
```

合成公式是 `out = new*α + orig*(1-α)`，且代码里对 `α == 0` 的像素
**强制直接取原图值**（`utils.paste_back` 中的 `hard_keep`）。
因此掩膜之外的像素在数学上逐字节等于输入，不存在"全局轻微色偏"这种问题。

模型也**只看到 ROI**，不接触全图，避免了整图重编码带来的隐性损失。

### 5.2 "笔画级掩膜"为什么重要

擦文字/水印时，若把整个矩形框都交给模型，框内原本真实的背景
（木纹、渐变、人脸的一部分）会被一并抹掉再"编"回来。
本项目在框内做 2-means 聚类，只把**属于笔画的像素**标为待修复，
背景像素原样保留。擦除面积通常能缩小到框面积的 15~25%，
结果自然度显著提升。

### 5.3 文字风格是怎么"保持"的

把"风格"拆成七个可测量的量，逐个从原图估计：

| 维度 | 估计方法 |
|------|----------|
| 前景色 | ROI 内 2-means，取像素较少的一类（少数类=笔画） |
| 背景色 | 2-means 的多数类 |
| 字号 | 笔画掩膜的真实墨迹高度（不是 OCR 框高，后者含内边距） |
| 粗细 | 笔画面积 / 墨迹外接框面积 → 占空比阈值判断 |
| 倾斜 | OCR 四点多边形上边缘的方位角 |
| 透视 | 直接用四点做单应性变换，天然携带透视 |
| 质感 | 拉普拉斯方差 → 高斯 σ；MAD → 噪声 σ，二者施加回新字 |

渲染时用 **4× 超采样 + LANCZOS 下采样**，让笔画抗锯齿接近原生排版。

> 说明：这条路线是**参数化**的，可解释、可调、CPU 可跑。
> 若追求极致，可在 `core/text_edit.py` 里接入 SRNet / AnyText 等
> 场景文字编辑模型——接口已经预留（`replace_text` 的 `style` 参数）。

### 5.4 PDF 如何"保留原清晰度"

固定 DPI 光栅化是个陷阱：一份内嵌 600 DPI 扫描图的 PDF，用 150 DPI 渲染
就是**降采样到 1/4**。本项目按页计算原生分辨率：

```
native_dpi = 页内最大位图的像素宽 ÷ (页宽pt ÷ 72)
render_dpi = clamp(max(用户设定, native_dpi), 72, 600)
```

回写时页面尺寸严格沿用原页的 pt 值，图片按原像素数嵌入。

### 5.5 水印检测的能力边界（请务必知悉）

单张任意图片上的**盲水印检测本身是欠定问题**——学术上通常需要
"同一水印的多张图"来联立求解水印的 α 与 RGB。所以本项目采用
**四线索投票 + 人工确认**：

1. 重复文本（OCR 同一串文字出现 ≥2 次）
2. 版权符号与域名关键词（© ® ™ / `www.` / `.com` / "版权"…）
3. 半透明细笔画（形态学 tophat/blackhat 的**带通**响应）
4. 周期性平铺（高通残差的 FFT 自相关峰）

**程序永远不会自动直接擦除**。误擦一张照片里的真实招牌，
代价远大于让用户多点一下鼠标。检测不到时，手动框选的效果完全一样。

---

## 六、性能参考

| 场景 | CPU（i7-12700） | GPU（RTX 3060） |
|------|-----------------|-----------------|
| 512×512 ROI · LaMa | 1.5 ~ 3 s | 0.1 ~ 0.3 s |
| 1600×1600 ROI · LaMa | 8 ~ 20 s | 0.5 ~ 1 s |
| OCR 单页 A4 300DPI | 2 ~ 6 s | 0.8 ~ 2 s |
| SD Inpainting 512² · 25 步 | 3 ~ 8 min（不推荐） | 3 ~ 6 s |

提速建议：
- 调小「上下文比例」（0.7 → 0.4）
- 保持「连通域拆分」开启（默认开），零散小目标会被分开处理
- 大图先用「1:1」定位，只涂真正需要的区域

---

## 七、常见问题

**Q：启动报 `No module named 'PyQt5'`**
A：`pip install PyQt5`。若是 Linux 无显示服务器，需要 `apt install libxcb-xinerama0`。

**Q：状态栏显示「OpenCV 传统修复（未加载 LaMa）」**
A：说明 `big-lama.pt` 没下下来。运行 `python main.py --predownload` 看具体报错，
或手动下载后放进 `./models_zoo/`。

**Q：点「识别文字」没反应/报错**
A：没装 OCR。`pip install paddlepaddle paddleocr` 或 `pip install easyocr`。
装完重启程序。

**Q：修改后的文字字体和原图不完全一样**
A：程序只能从本机已安装字体里挑最接近的。若知道原字体，
在「修改文字」对话框的「字体」下拉里手动指定即可。

**Q：擦除后留下一圈淡淡的痕迹**
A：把「选区膨胀」调大（4 → 8），通常是物体的抗锯齿边缘或投影没被覆盖。

**Q：处理超大图（>8000px）内存吃紧**
A：ROI 机制已经把峰值内存限制在 ROI 大小，但撤销栈会缓存 20 份历史。
如需降低占用，可修改 `ui.py` 中 `_set_cur_image` 里的 `> 20` 阈值。

---

## 八、许可与合规提醒

- 本工具的图像修复能力**不得**用于伪造证件、篡改合同凭证、
  去除他人作品的版权标识等违法用途。
- LaMa / Stable Diffusion / PaddleOCR 各自遵循其原始开源许可，
  商用前请自行核对。

---

## 九、打包与发布（PyInstaller / GitHub Release）

仓库已内置可复用的打包配置，无需从零写 spec。

### 本地打包（Windows）

```bash
pip install pyinstaller
pyinstaller build.spec          # 产物在 dist/smart_image_editor/
# 或用一键脚本（自动装 pyinstaller）：
build_exe.bat
```

`build.spec` 的关键策略：

- **排除** `torch` / `paddle` / `diffusers` 等重型库 → EXE 体积小、且避开
  Windows 上 `c10.dll` 的 VC++ 运行时依赖问题（WinError 1114）。
- **内置** LaMa-ONNX（`onnxruntime`）+ OpenCV + PyQt5 + PyMuPDF → 开箱即用。
- 冻结后模型缓存放到 `%APPDATA%/smart_image_editor/models`，不会被临时目录清除。

### 自动发布（GitHub Actions）

推送一个 **tag** 即可触发 `.github/workflows/build.yml`：

```bash
git tag v1.0.0
git push origin v1.0.0        # 自动构建 EXE 并作为 Release 附件
```

该 workflow 在 Windows runner 上构建，并将 `smart_image_editor-<tag>-win64.zip`
附加到自动创建的 Release 中——你只需打 tag，无需手动打包。

> 想调整打包内容（例如把 torch 也打进去以支持 LaMa-TorchScript 后端），
> 编辑 `build.spec` 的 `excludes` / `hiddenimports` 即可。
