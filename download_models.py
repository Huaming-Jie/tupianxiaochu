# -*- coding: utf-8 -*-
"""
download_models.py —— 模型权重自动下载（离线/首次部署专用）
================================================================================
程序在「首次使用修复功能」时会自动按需下载对应权重，本脚本只是把这一过程
提前、显式地跑一遍，方便：

* 在内网/离线机器上先在有网的机器下好，再拷贝过去
* CI / 打包前预拉取权重
* 排查下载源是否可达

用法::

    python download_models.py            # 下载全部可用权重
    python download_models.py --only onnx   # 只下载 LaMa ONNX（约 200MB，无需 torch）
    python download_models.py --only ts     # 只下载 LaMa TorchScript（需本机已装 torch）

权重默认存放于 ``models_zoo/``（可用环境变量 SIE_MODEL_DIR 更改）。
"""
from __future__ import annotations

import argparse
import sys

from models.downloader import MODEL_REGISTRY, get_model_path

# 两个权重 key：ts = TorchScript（需 torch），onnx = ONNX（无需 torch）
ALL_KEYS = ["lama_ts", "lama_onnx"]


def main() -> int:
    parser = argparse.ArgumentParser(description="智能图像编辑器 · 模型权重下载")
    parser.add_argument("--only", choices=["ts", "onnx", "all"], default="all",
                        help="只下载指定权重：ts=TorchScript, onnx=ONNX, all=全部")
    parser.add_argument("--no-ts", action="store_true",
                        help="跳过 TorchScript（本机没装 torch 时用）")
    args = parser.parse_args()

    keys = []
    if args.only in ("ts", "all") and not args.no_ts:
        keys.append("lama_ts")
    if args.only in ("onnx", "all"):
        keys.append("lama_onnx")

    if not keys:
        print("没有需要下载的权重。")
        return 0

    ok = True
    for k in keys:
        spec = MODEL_REGISTRY[k]
        print(f"\n>>> 下载 {spec.filename}（{spec.size_hint}）— {spec.desc}")
        try:
            path = get_model_path(k, progress=lambda c, t, m: print(f"\r  {m}", end=""))
            print()
        except Exception as e:  # noqa: BLE001
            print(f"\n  ✗ 下载失败：{e}")
            path = None
        if path:
            print(f"  ✓ 已就绪：{path}")
        else:
            ok = False
            print(f"  ✗ {spec.filename} 下载失败，请参考上方提示手动下载。")

    print("\n" + ("全部权重就绪 ✅" if ok else "部分权重缺失 ⚠️"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
