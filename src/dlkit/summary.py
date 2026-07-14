"""
dlkit.summary
=============

轻量级标量记录器 ``Summary``，用于在训练过程中收集标量指标并保存为 JSON。

支持自动路径推导、关键字参数快捷记录、自动保存等功能。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["Summary"]


class Summary:
    """标量指标记录器。

    在训练过程中收集标量值（如 loss、accuracy），保存为 JSON 文件。

    **文件路径推导**（优先级从高到低）：

    1. 直接传入 ``root`` 参数。
    2. 从 ``args`` 中读取 ``record_dir``、``record_tag``、``platform``，
       拼接为 ``[record_dir]/[主调文件名]_[record_tag]_[platform].json``。
    3. 全部缺省：默认保存到 ``./records/[主调文件名].json``。

    **两种记录方式**：

    * ``log(**kwargs)`` — 记录多个指标，参数名即指标名，自动递增 step。单条记录同样适用。

    :param root: JSON 文件完整路径。若为 ``None`` 则从 ``args`` 或默认值推导。
    :param args: ``StandardTrainingPipeline.init()`` 返回的 args 对象，
                 从中读取 ``record_dir``、``record_tag``、``platform`` 等关键字。
    :param auto_save: 是否在每次 ``log()`` 后自动保存到文件。
    """

    # JSON 中 step 和 value 的键名
    _STEP_KEY: str = "step"
    _VALUE_KEY: str = "value"

    def __init__(
        self,
        root: Optional[str] = None,
        args: Any = None,
        auto_save: bool = True,
    ):
        self.root = self._resolve_root(root, args)
        self.auto_save = auto_save
        self.records: Dict[str, Dict[str, list]] = {}
        self._step: int = 0

        # 确保保存目录存在
        os.makedirs(os.path.dirname(self.root) or ".", exist_ok=True)

        # 如果文件已存在，加载历史记录
        self._load_existing()

    # ------------------------------------------------------------------
    #  路径推导
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_root(root: Optional[str], args: Any) -> str:
        """推导 JSON 文件路径。

        优先级：``root`` 参数 > ``args`` 中的关键字 > 默认值 ``./records/``。
        """
        if root is not None:
            return root

        record_dir = None
        record_tag = None
        platform = None

        if args is not None:
            record_dir = getattr(args, "record_dir", None)
            record_tag = getattr(args, "record_tag", None)
            platform = getattr(args, "platform", None)

        try:
            from .utils import get_caller_filename
        except ImportError:
            from dlkit.utils import get_caller_filename
        caller_name = get_caller_filename(stem=True)

        parts = [caller_name]
        if record_tag:
            parts.append(record_tag)
        if platform:
            parts.append(platform)
        filename = "_".join(parts) + ".json"

        save_dir = Path(record_dir) if record_dir else Path.cwd() / "records"
        return str(save_dir / filename)

    # ------------------------------------------------------------------
    #  加载已有记录
    # ------------------------------------------------------------------
    def _load_existing(self):
        """如果 JSON 文件已存在，加载历史记录以便追加。"""
        if not os.path.exists(self.root):
            return
        try:
            with open(self.root, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[Summary] 无法加载已有记录: {e}")
            return

        if not isinstance(data, dict):
            return

        max_step = 0
        for name, record in data.items():
            if isinstance(record, dict) and self._STEP_KEY in record:
                self.records[name] = record
                if record[self._STEP_KEY]:
                    max_step = max(max_step, max(record[self._STEP_KEY]))
        self._step = max_step + 1
        if self.records:
            print(f"[Summary] Loaded existing records from: [{self.root}] (resumed at step {self._step})")

    # ------------------------------------------------------------------
    #  记录
    # ------------------------------------------------------------------
    def log(self, step: Optional[int] = None, **kwargs):
        """记录多个标量：将关键字参数名作为指标名、参数值作为值记录。

        用法::

            summary = Summary(root='./records/train.json')
            summary.log(train_loss=0.45, val_acc=0.92, lr=0.01)
            summary.log(loss=0.5)  # 单条记录同样适用

        :param step: 步数。若为 ``None`` 则自动递增。
        :param kwargs: 指标名=值 的键值对。
        """
        if not kwargs:
            return

        s = step if step is not None else self._step
        for name, value in kwargs.items():
            if name not in self.records:
                self.records[name] = {self._STEP_KEY: [], self._VALUE_KEY: []}
            self.records[name][self._STEP_KEY].append(s)
            self.records[name][self._VALUE_KEY].append(value)
        if step is None:
            self._step += 1

        if self.auto_save:
            self.save()

    # ------------------------------------------------------------------
    #  保存与加载
    # ------------------------------------------------------------------
    def save(self):
        """将当前所有记录保存到 JSON 文件（原子写入）。"""
        tmp_path = self.root + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, self.root)  # 原子替换

    def load(self):
        """重新从文件加载记录（覆盖内存中的记录）。"""
        self.records.clear()
        self._load_existing()

    # ------------------------------------------------------------------
    #  查询
    # ------------------------------------------------------------------
    def get(self, name: Optional[str] = None) -> Dict:
        """获取指定指标或全部记录。

        :param name: 指标名称。若为 ``None`` 则返回全部记录。
        :raises KeyError: 当 ``name`` 不存在时。
        """
        if name is not None:
            if name not in self.records:
                raise KeyError(f'"{name}" is not recorded')
            return self.records[name]
        return self.records

    def keys(self) -> List[str]:
        """返回所有已记录的指标名称列表。"""
        return list(self.records.keys())

    @property
    def step(self) -> int:
        """当前步数。"""
        return self._step

    @step.setter
    def step(self, value: int):
        """设置当前步数。"""
        self._step = value

    # ------------------------------------------------------------------
    #  重置
    # ------------------------------------------------------------------
    def reset(self):
        """清空所有记录并重置步数。"""
        self.records.clear()
        self._step = 0
