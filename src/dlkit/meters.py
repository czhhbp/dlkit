"""
dlkit.meters
============

轻量级标量计量器与计时器。
"""

from __future__ import annotations

import time

__all__ = ["ScalarMeter", "TimeMeter"]


class ScalarMeter:
    """自动累加标量并计算均值/总和的计量器。

    :param reduction: 传入 ``value`` 的归约方式。

        * ``'mean'`` — ``value`` 是一批数据的均值，内部按 ``value * num`` 累加。
        * ``'sum'``  — ``value`` 是一批数据的总和，直接累加。
    """

    def __init__(self, reduction: str = "mean"):
        assert reduction in ("mean", "sum"), "reduction must be 'mean' or 'sum'"
        self.reduction = reduction
        self.reset()

    def reset(self):
        """重置累加器。"""
        self._sum: float = 0.0
        self._count: float = 0.0

    def update(self, value: float, num: int = 1):
        """接收新数据。

        :param value: 经过 reduction 后的数值标量。
        :param num: 该批数据的数量。
        """
        self._count += num
        if self.reduction == "sum":
            self._sum += value
        else:
            self._sum += value * num

    @property
    def avg(self) -> float:
        """返回当前均值，无数据时返回 0.0。"""
        return self._sum / self._count if self._count > 0 else 0.0

    @property
    def sum(self) -> float:
        """返回当前总和。"""
        return self._sum

    @property
    def count(self) -> float:
        """返回当前累计数据量。"""
        return self._count


class TimeMeter:
    """计时器，支持分段计时与总计时。

    修复说明：原版 ``start_time`` 未初始化导致 ``total`` 属性崩溃，
    此版本在构造时自动记录起始时间。

    用法::

        timer = TimeMeter()
        # ... 执行代码 ...
        timer.update()        # 记录一个时间点
        print(timer.cost)     # 距上一次 update 的耗时（分钟）
        print(timer.total)    # 从开始到现在的总耗时（分钟）
    """

    def __init__(self):
        now = time.time()
        self.start_time: float = now
        self.last_point: float = now
        self.last_time: float = 0.0

    def reset(self):
        """重置计时器（重新开始计时）。"""
        now = time.time()
        self.start_time = now
        self.last_point = now
        self.last_time = 0.0

    def update(self):
        """记录一个时间点，更新分段耗时。"""
        t = time.time()
        self.last_time = t - self.last_point
        self.last_point = t

    @property
    def total(self) -> float:
        """从开始到现在的总耗时（分钟，保留 4 位小数）。"""
        return round((time.time() - self.start_time) / 60, 4)

    @property
    def cost(self) -> float:
        """与上一次 ``update()`` 之间的持续时间（分钟，保留 4 位小数）。"""
        return round(self.last_time / 60, 4)
