# -*- coding: utf-8 -*-
"""
models/downloader.py —— 模型权重下载器
================================================================================
职责
----
* 维护一张"模型注册表"，每个模型有 **多个候选下载源**（国内镜像优先）
* 带 **断点续传**（HTTP Range）、**进度回调**、**MD5 校验** 的稳健下载
* 为 HuggingFace 生态（diffusers / transformers）统一设置 hf-mirror 端点

为什么要多源候选？
------------------
国内直连 huggingface.co / github.com 的成功率不稳定。策略是：
hf-mirror 镜像 → GitHub Release → ghproxy 代理 → 官方源，逐个尝试，
任何一个成功即可。全部失败时给出**明确的手动下载指引**而不是抛一个裸异常。
"""

from __future__ import annotations

import hashlib
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from utils import LOG, MODEL_DIR, ensure_dir, human_size

# --------------------------------------------------------------------------- #
# 1. HuggingFace 镜像设置
# --------------------------------------------------------------------------- #

HF_MIRROR = os.environ.get("SIE_HF_ENDPOINT", "https://hf-mirror.com")


def set_hf_mirror(endpoint: str = HF_MIRROR) -> None:
    """设置 HF_ENDPOINT 环境变量。

    必须在 ``import huggingface_hub / diffusers`` **之前** 调用才生效，
    因此 main.py 会在最开头调用它。
    """
    os.environ.setdefault("HF_ENDPOINT", endpoint)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    # 把 HF 缓存也放进项目目录，便于整体打包/迁移
    os.environ.setdefault("HF_HOME", os.path.join(MODEL_DIR, "hf_home"))
    LOG.info("HuggingFace 端点已设置为 %s", endpoint)


# --------------------------------------------------------------------------- #
# 2. 模型注册表
# --------------------------------------------------------------------------- #

@dataclass
class ModelSpec:
    """一个可下载权重文件的描述。"""
    key: str                              # 唯一标识
    filename: str                         # 落盘文件名
    urls: List[str] = field(default_factory=list)   # 候选下载地址（按优先级）
    md5: Optional[str] = None             # 可选校验码
    size_hint: str = "未知"                # 用于 UI 提示
    desc: str = ""

    @property
    def local_path(self) -> str:
        return os.path.join(MODEL_DIR, self.filename)


#: 全局模型注册表
MODEL_REGISTRY: dict[str, ModelSpec] = {
    # ---------------- LaMa 大掩膜修复（TorchScript，动态尺寸，CPU 可跑） ----
    "lama_ts": ModelSpec(
        key="lama_ts",
        filename="big-lama.pt",
        urls=[
            # ghproxy 优先：国内对 GitHub Release 的稳定代理，实测可达且快速
            "https://ghproxy.net/https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
            "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
            "https://hf-mirror.com/Carve/LaMa-ONNX/resolve/main/big-lama.pt",
            "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/big-lama.pt",
        ],
        md5=None,  # 先放行下载；下载后用 torch.jit.load 校验有效性，再把真实 md5 写回
        size_hint="≈196 MB",
        desc="LaMa 大掩膜修复模型（TorchScript）——物体/水印/文字消除主力",
    ),
    # ---------------- LaMa ONNX 版（无 torch 环境时的备胎） ----------------
    "lama_onnx": ModelSpec(
        key="lama_onnx",
        filename="lama_fp32.onnx",
        urls=[
            "https://ghproxy.net/https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
            "https://hf-mirror.com/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
            "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
        ],
        size_hint="≈206 MB",
        desc="LaMa ONNX 版——仅在 PyTorch 不可用时启用",
    ),
}


# --------------------------------------------------------------------------- #
# 3. 下载实现
# --------------------------------------------------------------------------- #

ProgressCb = Callable[[int, int, str], None]   # (已下载, 总量, 提示文本)


def _md5_of(path: str, chunk: int = 1 << 20) -> str:
    """计算文件 MD5。"""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _download_one(url: str, dest: str, progress: Optional[ProgressCb] = None,
                  timeout: int = 30, retries: int = 2) -> bool:
    """从单个 URL 下载到 dest（支持断点续传），成功返回 True。"""
    try:
        import requests
    except ImportError:
        LOG.error("缺少 requests 库，无法自动下载模型：pip install requests")
        return False

    tmp = dest + ".part"
    headers = {"User-Agent": "smart-image-editor/1.0"}

    for attempt in range(retries + 1):
        try:
            downloaded = os.path.getsize(tmp) if os.path.exists(tmp) else 0
            if downloaded:
                headers["Range"] = f"bytes={downloaded}-"
            else:
                headers.pop("Range", None)

            with requests.get(url, stream=True, timeout=timeout,
                              headers=headers, allow_redirects=True) as r:
                if r.status_code == 416:      # 已经下完
                    break
                if r.status_code not in (200, 206):
                    LOG.warning("源返回 HTTP %s：%s", r.status_code, url)
                    return False

                total = int(r.headers.get("Content-Length", 0)) + \
                    (downloaded if r.status_code == 206 else 0)

                mode = "ab" if r.status_code == 206 and downloaded else "wb"
                if mode == "wb":
                    downloaded = 0

                last_report = 0.0
                with open(tmp, mode) as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        now = time.time()
                        if progress and now - last_report > 0.2:
                            last_report = now
                            progress(downloaded, total,
                                     f"下载中 {human_size(downloaded)}/{human_size(total)}")
            break
        except Exception as e:                       # 网络抖动 → 重试
            LOG.warning("下载失败(%s/%s)：%s", attempt + 1, retries + 1, e)
            time.sleep(1.5 * (attempt + 1))
    else:
        return False

    if not os.path.exists(tmp) or os.path.getsize(tmp) < 1024:
        return False
    shutil.move(tmp, dest)
    return True


def get_model_path(key: str, progress: Optional[ProgressCb] = None,
                   auto_download: bool = True) -> Optional[str]:
    """获取模型本地路径；不存在时按注册表候选源依次尝试下载。

    返回
    ----
    成功 → 本地绝对路径；失败 → None（调用方应降级到备用算法，而不是崩溃）
    """
    spec = MODEL_REGISTRY.get(key)
    if spec is None:
        LOG.error("未知模型 key：%s", key)
        return None

    ensure_dir(MODEL_DIR)
    path = spec.local_path

    # 已存在则校验后直接返回
    if os.path.isfile(path) and os.path.getsize(path) > 1024:
        if spec.md5:
            if _md5_of(path) == spec.md5:
                return path
            LOG.warning("模型 %s MD5 校验不通过，将重新下载", spec.filename)
            os.remove(path)
        else:
            return path

    if not auto_download:
        return None

    LOG.info("开始下载模型 %s (%s)", spec.filename, spec.size_hint)
    for i, url in enumerate(spec.urls, 1):
        if progress:
            progress(0, 0, f"尝试源 {i}/{len(spec.urls)}…")
        LOG.info("  → 源 %s/%s: %s", i, len(spec.urls), url)
        if _download_one(url, path, progress):
            if spec.md5 and _md5_of(path) != spec.md5:
                LOG.warning("  校验失败，换下一个源")
                os.remove(path)
                continue
            LOG.info("模型已就绪：%s", path)
            return path

    LOG.error(
        "\n" + "=" * 72 +
        f"\n模型 {spec.filename} 自动下载失败。请手动下载后放到：\n  {path}\n"
        f"可用地址：\n  " + "\n  ".join(spec.urls) +
        "\n" + "=" * 72
    )
    return None


def model_exists(key: str) -> bool:
    """仅检查本地是否已有该模型（不触发下载）。"""
    spec = MODEL_REGISTRY.get(key)
    return bool(spec and os.path.isfile(spec.local_path)
                and os.path.getsize(spec.local_path) > 1024)
