"""
dlkit.pipeline
==============

标准化的深度学习实验流水线框架 ``StandardTrainingPipeline``。

负责整合配置解析、设备分配（单卡/多卡/CPU）、随机种子设定、
工作目录初始化、代码快照备份以及全局日志系统。

修复说明：
  - ``init(enable_directory=False)`` 时 ``self.print`` 未定义的问题已修复。
  - ``_init_seed`` 中使用 ``manual_seed_all`` 覆盖所有 GPU。
"""

from __future__ import annotations

import logging
import random
import re
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from .argparse_utils import GroupArgparse
from .utils import get_caller_filename

__all__ = ["StandardTrainingPipeline", "print_log"]


# ------------------------------------------------------------------
#  模块级结构化日志函数
#  在 STP.init() 后，任何模块 import 此函数即可使用
# ------------------------------------------------------------------
def print_log(*args, **kwargs):
    """结构化日志输出，等效于 ``pipeline.print_log()``。

    在 STP 初始化后，任何模块中调用此函数都会同时输出到控制台和日志文件。

    用法::

        from dlkit.pipeline import print_log
        print_log("Epoch 1 finished", loss=0.45, acc="92%")
        # 输出: 2026/07/13 02:30:00 Epoch 1 finished loss=0.45 acc=92%
    """
    msg = " ".join(str(arg) for arg in args)
    if kwargs:
        msg += " " + " ".join(f"{k}={v}" for k, v in kwargs.items())
    logging.info(msg.strip())


class StandardTrainingPipeline:
    """标准化的深度学习实验流水线框架。

    负责整合配置解析、设备分配（单卡/多卡/CPU）、随机种子设定、
    工作目录初始化、代码快照备份以及全局日志系统。

    实例此对象的目录将作为顶级目录，故务必将 main 程序放在实际顶层目录下。
    依据命令行 args 参数来构建初始化流程。

    ====================================================================

    **【保留参数名（Reserved Config Keys）】**

    以下参数名被流水线内部使用，在 base_config / YAML / 命令行中配置后会被自动识别。
    请勿将这些名称用于其他用途，否则可能被覆盖。

    **[计算与环境控制]**

    +-------------------+------------------+----------+-------------------------------------------+
    | 参数名            | 类型             | 默认值   | 说明                                      |
    +===================+==================+==========+===========================================+
    | ``gpu``           | bool / int       | True     | 是否使用 GPU。``False`` 强制 CPU；         |
    |                   |                  |          | 传入 int 指定单卡卡号。                    |
    +-------------------+------------------+----------+-------------------------------------------+
    | ``multi_gpus``    | bool / str / list| False    | 多卡并行开关。``True`` 使用全部可用卡；    |
    |                   |                  |          | ``"0,1"`` 或 ``[0, 1]`` 指定卡号。         |
    +-------------------+------------------+----------+-------------------------------------------+
    | ``seed``          | int              | 自动生成 | 全局随机种子。控制 numpy/random/torch。    |
    |                   |                  |          | 不提供则按时间戳自动生成。                 |
    +-------------------+------------------+----------+-------------------------------------------+
    | ``device``        | str              | 自动设置 | 计算设备标识（如 ``'cuda:0'``、``'cpu'``）。|
    |                   |                  |          | 由 ``_init_device`` 自动写入，无需手动配置。|
    +-------------------+------------------+----------+-------------------------------------------+
    | ``device_ids``    | list[int] / None | None     | 多卡设备 ID 列表。单卡或 CPU 时为 None。   |
    |                   |                  |          | 由 ``_init_device`` 自动写入。             |
    +-------------------+------------------+----------+-------------------------------------------+

    **[工作目录与备份]**

    +-------------------+--------+---------------------+------------------------------------------+
    | 参数名            | 类型   | 默认值              | 说明                                     |
    +===================+========+=====================+==========================================+
    | ``workspace``     | str    | ``'experiments'``   | Level 1 工作区根目录名（所有实验的根）。   |
    +-------------------+--------+---------------------+------------------------------------------+
    | ``project_name``  | str    | 主调脚本名          | Level 2 项目/任务目录名。                 |
    |                   |        | (如 ``'train'``)    | 不提供则自动获取主调文件名。               |
    +-------------------+--------+---------------------+------------------------------------------+
    | ``save_dir``      | str    | 自动生成            | 完整实验目录路径。                         |
    |                   |        |                     | 由 ``_init_directory`` 自动写入。          |
    +-------------------+--------+---------------------+------------------------------------------+

    **[平台标识]**

    +-------------------+--------+----------+------------------------------------------+
    | 参数名            | 类型   | 默认值   | 说明                                     |
    +===================+========+==========+==========================================+
    | ``platform``      | str    | ``''``   | 当前运行所在的平台名称（如 ``'6001'``）。  |
    |                   |        |          | 用于 ``gen_csv_path`` 自动生成 CSV 文件名。|
    +-------------------+--------+----------+------------------------------------------+

    **[CSV 记录]**

    +-------------------+--------+---------------------+------------------------------------------+
    | 参数名            | 类型   | 默认值              | 说明                                     |
    +===================+========+=====================+==========================================+
    | ``csv_dir``       | str    | ``'./csv'``         | CSV 文件保存目录。                         |
    |                   |        |                     | ``CSVExpTracker`` 未指定 ``root`` 时使用。 |
    +-------------------+--------+---------------------+------------------------------------------+
    | ``csv_tag``       | str    | ``''``              | CSV 文件名标签（如 ``'search'``）。         |
    |                   |        |                     | 用于区分不同实验阶段的 CSV 文件。           |
    +-------------------+--------+---------------------+------------------------------------------+

    **【显式关键字参数 vs base_config】**

    上表中的保留参数除了在 ``base_config`` / YAML / 命令行中配置外，
    也可以直接作为 ``__init__`` 的关键字参数传入，获得 IDE 自动补全与类型提示::

        # 直接使用关键字参数（推荐，最直观）
        pipeline = StandardTrainingPipeline(gpu=0, platform='6001', seed=42)

        # 等价于在 base_config 中配置
        pipeline = StandardTrainingPipeline(base_config={'gpu': 0, 'platform': '6001', 'seed': 42})

    优先级（从高到低）：命令行参数 > ``base_config`` > 显式关键字参数 > YAML > 内置默认值。

    :param base_config: 基础配置字典。
    :param yaml_path: YAML 配置文件路径。
    :param gpu: 是否使用 GPU。``False`` 强制 CPU；传入 int 指定单卡卡号。默认 ``None`` (即 ``True``)。
    :param multi_gpus: 多卡并行开关。``True`` 使用全部可用卡；``"0,1"`` 或 ``[0, 1]`` 指定卡号。默认 ``None`` (即 ``False``)。
    :param seed: 全局随机种子。``None`` 则自动生成。
    :param platform: 当前运行所在的平台名称（如 ``'6001'``），用于 ``gen_csv_path``。默认 ``None`` (即空字符串)。
    :param workspace: 工作区根目录名（所有实验的根目录）。默认 ``None`` (即 ``'experiments'``)。
    :param project_name: 项目/任务目录名。``None`` 则自动获取主调文件名。
    :param csv_dir: CSV 文件保存目录。``CSVExpTracker`` 未指定 ``root`` 时从此读取。默认 ``None`` (即 ``'./csv'``)。
    :param csv_tag: CSV 文件名标签（如 ``'search'``），用于区分不同实验阶段。默认 ``None`` (即空字符串)。
    :param backup_suffixes: 额外需要备份的文件后缀列表。
    :param ignore_dirs: 额外需要忽略的目录名列表。
    """

    def __init__(
        self,
        base_config: Optional[Dict[str, Any]] = None,
        yaml_path: Optional[str] = None,
        *,
        gpu: Union[bool, int, None] = None,
        multi_gpus: Union[bool, str, List[int], None] = None,
        seed: Optional[int] = None,
        platform: Optional[str] = None,
        workspace: Optional[str] = None,
        project_name: Optional[str] = None,
        csv_dir: Optional[str] = None,
        csv_tag: Optional[str] = None,
        backup_suffixes: Optional[List[str]] = None,
        ignore_dirs: Optional[List[str]] = None,
    ):
        # 将显式关键字参数收集为保留配置（仅非 None 值），作为最低优先级的默认层
        reserved_config: Dict[str, Any] = {}
        if gpu is not None:
            reserved_config["gpu"] = gpu
        if multi_gpus is not None:
            reserved_config["multi_gpus"] = multi_gpus
        if seed is not None:
            reserved_config["seed"] = seed
        if platform is not None:
            reserved_config["platform"] = platform
        if workspace is not None:
            reserved_config["workspace"] = workspace
        if project_name is not None:
            reserved_config["project_name"] = project_name
        if csv_dir is not None:
            reserved_config["csv_dir"] = csv_dir
        if csv_tag is not None:
            reserved_config["csv_tag"] = csv_tag

        # 合并：base_config 覆盖 reserved_config，得到传给 GroupArgparse 的 base_config
        if base_config:
            merged_base = GroupArgparse._recursive_merge(
                base_config, reserved_config.copy()
            )
        else:
            merged_base = reserved_config if reserved_config else None

        self.gparser = GroupArgparse(base_config=merged_base, yaml_path=yaml_path)
        self.config: Dict[str, Any] = {}
        self.args: Any = None
        self.device: Optional[str] = None
        self.multi_gpus: bool = False
        self.device_ids: Optional[List[int]] = None
        self.save_dir: Path = Path("./")
        self.run_name: str = ""
        self.logger: Optional[logging.Logger] = None
        self.platform: str = ""  # 当前运行所在的平台名称，用于 gen_csv_path
        self.print: Callable = print  # 修复：默认设为 print，避免 enable_directory=False 时未定义

        _backup_suffixes = set(backup_suffixes or [])
        _ignore_dirs = set(ignore_dirs or [])
        self.backup_suffixes = _backup_suffixes | {
            "py",
            "yaml",
            "yml",
            "json",
            "sh",
            "bash",
        }
        self.ignore_dirs = _ignore_dirs | {
            ".git",
            "__pycache__",
            "experiments",
            "results",
            "data",
            "figures",
            "pictures",
            "venv",
            "env",
            ".idea",
            ".vscode",
        }
        # 默认种子池
        self.seed_pool = [
            11, 22, 33, 44, 55, 66, 77, 88, 99,
            111, 222, 333, 444, 555, 666, 777, 888, 999,
            1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999,
            11111, 22222, 33333, 44444, 55555, 66666, 77777, 88888, 99999,
        ]

    # ------------------------------------------------------------------
    #  核心初始化
    # ------------------------------------------------------------------
    def init(
        self, run_name: str = "", enable_directory: bool = True
    ) -> Tuple[Any, Optional[Dict[str, Any]]]:
        """执行核心初始化流程。

        :param run_name: 运行名称，会拼接到实验目录名中。
        :param enable_directory: 是否创建实验目录和日志。设为 ``False`` 则跳过。
        :return: ``(args, config)``
        """
        self.gparser.parse_args()
        self.config = self.gparser.config
        self.run_name = run_name
        self.platform = self.config.get("platform", "") or ""

        if enable_directory:
            self._init_directory(self.config)
            self._init_logger()

        # 自适应 print 方法
        self.print = self.print_log if self.logger else print

        self._init_device(self.config)
        self._init_seed(self.config)
        self.args = self.gparser.nested_args(
            self.config, export_dataclass=True, schema_path="args_dataclass.py"
        )
        return self.args, self.config

    # ------------------------------------------------------------------
    #  设备
    # ------------------------------------------------------------------
    def _init_device(self, cfg: Dict[str, Any]):
        """配置计算设备。"""
        use_gpu = cfg.get("gpu", True)
        multi_gpus = cfg.get("multi_gpus", False)

        if not torch.cuda.is_available() or use_gpu is False:
            self.device = "cpu"
            self.multi_gpus = False
            self.device_ids = None
        else:
            if multi_gpus:
                if isinstance(multi_gpus, str):
                    device_ids = [int(x) for x in multi_gpus.split(",")]
                elif isinstance(multi_gpus, (list, tuple)):
                    device_ids = [int(x) for x in multi_gpus]
                else:
                    device_ids = list(range(torch.cuda.device_count()))

                self.device = f"cuda:{device_ids[0]}"
                self.multi_gpus = True
                self.device_ids = device_ids
                cfg["gpu"] = device_ids[0]
            else:
                device_id = use_gpu if isinstance(use_gpu, int) else 0
                self.device = f"cuda:{device_id}"
                self.multi_gpus = False
                self.device_ids = None
                cfg["gpu"] = device_id

        cfg["device_ids"] = self.device_ids
        cfg["device"] = self.device
        cfg["multi_gpus"] = self.multi_gpus

    # ------------------------------------------------------------------
    #  随机种子
    # ------------------------------------------------------------------
    def _init_seed(self, cfg: Dict[str, Any]):
        """初始化随机种子。"""
        seed = cfg.get("seed")
        if seed is None:
            seed = int(time.time() * 1000) % 1_000_000

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.config["seed"] = seed
        self.print(f"seed is {seed}")

        if "cuda" in cfg["device"]:
            torch.cuda.manual_seed_all(seed)  # 修复：覆盖所有 GPU
            cudnn.deterministic = True
            cudnn.benchmark = False

    # ------------------------------------------------------------------
    #  实验目录
    # ------------------------------------------------------------------
    def _init_directory(self, cfg: Dict[str, Any]):
        """创建三级实验目录并递归备份代码，保存运行记录。

        三级目录结构::

            workspace / project_name / run_name
            (工作区)    (项目名)      (运行实例)

        * ``workspace``    — Level 1，所有实验的根目录（默认 ``'experiments'``）。
        * ``project_name`` — Level 2，项目/任务名（默认自动获取主调脚本名）。
        * ``run_name``     — Level 3，单次运行实例名（时间戳 + 用户传入的 run_name）。
        """
        # Level 1: 工作区根目录
        workspace_name = cfg.get("workspace", "experiments")
        workspace_name = re.sub(r'[\\/*?:"<>|]', "_", workspace_name)
        workspace = Path(workspace_name)

        # Level 2: 项目/任务名
        default_script = (
            Path(sys.argv[0]).stem if sys.argv and sys.argv[0] else "interactive_run"
        )
        project_name = cfg.get("project_name", default_script)
        project_name = re.sub(r'[\\/*?:"<>|]', "_", project_name)

        # Level 3: 单次运行实例
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.run_name = "_".join([self.run_name, str(timestamp)]).strip("_")
        self.run_name = re.sub(r'[\\/*?:"<>|]', "_", self.run_name)

        self.save_dir = workspace / project_name / self.run_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 创建 script_dir 并递归备份代码
        script_dir = self.save_dir / "scripts"
        script_dir.mkdir(parents=True, exist_ok=True)

        suffixes = [str(ext).strip() for ext in self.backup_suffixes]
        current_dir = Path.cwd()

        for ext in suffixes:
            for file_path in current_dir.rglob(f"*.{ext}"):
                if any(part in self.ignore_dirs for part in file_path.parts):
                    continue
                relative_path = file_path.relative_to(current_dir)
                target_path = script_dir / relative_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, target_path)

        cfg["save_dir"] = str(self.save_dir)

    # ------------------------------------------------------------------
    #  日志
    # ------------------------------------------------------------------
    def _init_logger(self):
        """配置并启动日志模块。

        FileHandler 挂在 root logger 上，因此任何模块中调用
        ``logging.info()`` 都会同时输出到控制台和日志文件。

        同时设置 ``sys.excepthook``，将未捕获的异常也写入日志文件，
        确保 ``nohup ... >/dev/null 2>&1`` 时仍可查看报错信息。
        """
        log_format = "%(asctime)s %(message)s"
        logging.basicConfig(
            stream=sys.stdout,
            level=logging.INFO,
            format=log_format,
            datefmt="%Y/%m/%d %H:%M:%S",
        )

        # 将 FileHandler 挂在 root logger，使所有模块的 logging.info() 都能写入日志文件
        root_logger = logging.getLogger()
        fh = logging.FileHandler(self.save_dir / "log.txt")
        fh.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(fh)

        self.logger = root_logger

        # 捕获未处理的异常并写入日志文件
        _original_excepthook = sys.excepthook

        def _log_excepthook(exc_type, exc_value, exc_tb):
            import traceback
            tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)
            root_logger.error("Unhandled exception:\n" + "".join(tb_lines).rstrip())
            _original_excepthook(exc_type, exc_value, exc_tb)

        sys.excepthook = _log_excepthook

    def print_log(self, *args, **kwargs):
        """结构化日志输出，同时写入控制台和日志文件。

        等效于模块级函数 ``print_log()``，后者可在任何模块中直接调用。
        """
        msg = " ".join(str(arg) for arg in args)
        if kwargs:
            msg += " " + " ".join(f"{k}={v}" for k, v in kwargs.items())
        logging.info(msg.strip())

    # ------------------------------------------------------------------
    #  重命名实验目录
    # ------------------------------------------------------------------
    def rename_run_name(self, suffix: str):
        """实验结束追加后缀（如重命名为 ``run_name_Acc_95.5``）。"""
        if self.logger:
            # 关闭并移除 FileHandler，保留 StreamHandler
            for handler in list(self.logger.handlers):
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                    self.logger.removeHandler(handler)

        new_path = self.save_dir.parent / f"{self.save_dir.name}_{suffix}"
        try:
            self.save_dir.rename(new_path)
            self.save_dir = new_path
            self.print(f"Experiment directory renamed to: {new_path.name}")
        except Exception as e:
            self.print(f"Failed to rename directory: {e}")

    # ------------------------------------------------------------------
    #  CSV 文件名生成
    # ------------------------------------------------------------------
    def gen_csv_path(self, tag: str = "", save_dir: Optional[str] = None) -> str:
        """自动生成 CSV 文件路径。

        文件名格式为 ``[主调文件名]_[tag]_[platform].csv``。

        * **主调文件名**：自动获取调用此方法的脚本名（不含扩展名），
          如 ``train.py`` -> ``train``。
        * **tag**：用户自定义的标签，用于区分不同实验阶段或配置
          （如 ``'search'``、``'finetune'``）。
        * **platform**：当前运行所在的平台名称（通过 config 的 ``platform`` 参数配置），
          用于多机实验结果合并。若未设置则省略该段。

        示例::

            # config 中设置了 platform='6001'
            csv_path = pipeline.gen_csv_path(tag='search')
            # -> './csv/train_search_6001.csv'

            # 未设置 platform
            csv_path = pipeline.gen_csv_path(tag='search')
            # -> './csv/train_search.csv'

            # 不传 tag
            csv_path = pipeline.gen_csv_path()
            # -> './csv/train_6001.csv' 或 './csv/train.csv'

        :param tag: 自定义标签（如 ``'search'``、``'finetune'``）。
        :param save_dir: CSV 文件存放目录。若为 ``None`` 则使用当前工作目录下的 ``csv`` 文件夹。
        :return: 完整的 CSV 文件路径字符串。
        """
        parts = [get_caller_filename(stem=True)]

        if tag:
            parts.append(tag)

        if self.platform:
            parts.append(self.platform)

        filename = "_".join(parts) + ".csv"

        if save_dir is not None:
            base = Path(save_dir)
        else:
            base = Path.cwd() / "csv"

        return str(base / filename)
