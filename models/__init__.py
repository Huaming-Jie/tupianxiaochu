# -*- coding: utf-8 -*-
"""
models 包 —— 深度学习模型的下载、加载与推理封装。

每个子模块只负责"把某个模型跑起来"，不掺杂业务逻辑；
业务编排统一放在 core/ 包中。这样换模型时改动面最小。
"""

from .downloader import MODEL_REGISTRY, get_model_path, set_hf_mirror  # noqa: F401

__all__ = ["MODEL_REGISTRY", "get_model_path", "set_hf_mirror"]
