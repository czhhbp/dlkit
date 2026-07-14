"""
dlkit
=====

Deep Learning experiment assistant toolkit.

一个用于深度学习实验的辅助工具包，提供：

- **配置解析**: ``GroupArgparse`` — 多层级分组命令行参数解析器
- **实验流水线**: ``StandardTrainingPipeline`` — 设备分配、种子设定、目录管理、日志、代码备份、CSV 文件名生成
- **计量器**: ``ScalarMeter``、``TimeMeter`` — 标量累加与计时
- **实验追踪**: ``CSVExpTracker`` — CSV 存储、多服务器聚合、查重、统计
- **标量记录**: ``Summary`` — JSON 格式标量指标记录
- **工具函数**: ``set_seed``、``count_parameters_in_MB``、``get_gpus_memory_info``、``get_caller_filename`` 等

快速开始::

    from dlkit import StandardTrainingPipeline

    pipeline = StandardTrainingPipeline(base_config={'gpu': 0, 'seed': 42, 'platform': '6001'})
    args, config = pipeline.init(run_name='my_exp')
    csv_path = pipeline.gen_csv_path(tag='search')
    pipeline.print('CSV path:', csv_path)
"""

from .argparse_utils import GroupArgparse
from .meters import ScalarMeter, TimeMeter
from .pipeline import StandardTrainingPipeline, print_log
from .summary import Summary
from .tracker import CSVExpTracker
from .utils import (
    count_parameters_in_MB,
    cur_time_str,
    get_caller_filename,
    get_gpus_memory_info,
    set_seed,
)

__version__ = "0.2.1"

__all__ = [
    # pipeline
    "StandardTrainingPipeline",
    "print_log",
    # argparse
    "GroupArgparse",
    # meters
    "ScalarMeter",
    "TimeMeter",
    # tracker
    "CSVExpTracker",
    # summary
    "Summary",
    # utils
    "cur_time_str",
    "get_caller_filename",
    "get_gpus_memory_info",
    "set_seed",
    "count_parameters_in_MB",
]
