"""
dlkit.utils
===========

独立的通用辅助函数集合。
"""

from __future__ import annotations

import inspect
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

__all__ = [
    "cur_time_str",
    "get_caller_filename",
    "get_gpus_memory_info",
    "set_seed",
    "count_parameters_in_MB",
]


def cur_time_str() -> str:
    """返回当前时间的格式化字符串 ``%Y%m%d-%H%M%S``。"""
    return time.strftime("%Y%m%d-%H%M%S")


def get_caller_filename(stem: bool = True) -> str:
    """获取主调文件的文件名。

    通过回溯调用栈，找到第一个不属于 ``dlkit`` 包的调用帧，
    返回该帧对应的脚本文件名。

    :param stem: 若为 ``True``（默认），返回不含扩展名的文件名（如 ``'train'``）；
                 若为 ``False``，返回完整文件名（如 ``'train.py'``）。
    :return: 主调文件名。若在交互式环境中无法确定，返回 ``'interactive_run'``。

    示例::

        # 在 train.py 中调用
        from dlkit import get_caller_filename
        print(get_caller_filename())       # 'train'
        print(get_caller_filename(stem=False))  # 'train.py'
    """
    frame = inspect.currentframe()
    try:
        # 向上回溯，跳过 dlkit 包内部的帧
        caller = frame.f_back if frame is not None else None
        while caller is not None:
            filename = caller.f_code.co_filename
            # 判断是否属于 dlkit 包内部
            normalized = filename.replace("\\", "/")
            if "/dlkit/" not in normalized and "dlkit" not in Path(filename).name:
                p = Path(filename)
                return p.stem if stem else p.name
            caller = caller.f_back
    finally:
        # 避免引用循环
        del frame
    return "interactive_run"


def get_gpus_memory_info():
    """获取各 GPU 的可用显存信息。

    通过解析 ``nvidia-smi`` 输出，返回空闲显存最大的 GPU 编号及各卡可用显存列表。

    :return: ``(best_gpu_id, memory_available_list)``
    :rtype: tuple[int, list[int]]
    :raises RuntimeError: 当系统没有 NVIDIA GPU 或 nvidia-smi 不可用时。
    """
    rst = subprocess.run(
        "nvidia-smi -q -d Memory",
        stdout=subprocess.PIPE,
        shell=True,
    ).stdout.decode("utf-8")
    rst = rst.strip().split("\n")
    memory_available = [
        int(line.split(":")[1].split(" ")[1]) for line in rst if "Free" in line
    ][::2]
    if not memory_available:
        raise RuntimeError("未检测到可用的 GPU 显存信息，请确认 nvidia-smi 是否可用。")
    best_id = int(np.argmax(memory_available))
    return best_id, memory_available


def set_seed(seed: int | None = None) -> int:
    """统一设置 ``random``、``numpy``、``torch``（CPU + CUDA）的随机种子。

    :param seed: 随机种子。若为 ``None`` 则自动生成。
    :return: 实际使用的种子值。
    """
    if seed is None:
        seed = np.random.randint(1_000_000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 修复：使用 manual_seed_all 覆盖所有 GPU
    return seed


def count_parameters_in_MB(model: torch.nn.Module) -> float:
    """统计模型可训练参数量（单位：百万）。

    自动排除名称中包含 ``"aux"`` 的辅助参数。

    :param model: PyTorch 模型。
    :return: 参数量（MB，即百万）。
    """
    return (
        sum(
            p.numel()
            for name, p in model.named_parameters()
            if "aux" not in name
        )
        / 1e6
    )
