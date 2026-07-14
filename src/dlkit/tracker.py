"""
dlkit.tracker
=============

统一的实验记录管理器 ``CSVExpTracker``。

提供 CSV 存储、多平台结果聚合、自动统计计算、动态查重、
列名自动推断、新列自动迁移等功能。

两种写入方式：

* ``log(**kwargs)`` — 快捷写入，参数名即列名，无需预先设置列结构。
* ``save_csv(dict_or_list)`` — 字典写入，需预先设置 ``x_columns`` / ``y_columns``。
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

__all__ = ["CSVExpTracker"]


class CSVExpTracker:
    """统一的实验记录管理器。

    集成了配置/结果的 CSV 存储、多平台结果聚合、自动统计计算、
    动态查重、列名自动推断以及新列自动迁移功能。

    **CSV 文件路径推导**（优先级从高到低）：

    1. 直接传入 ``root`` 参数。
    2. 从 ``args`` 中读取 ``csv_dir``、``csv_tag``、``platform``，
       拼接为 ``[csv_dir]/[主调文件名]_[csv_tag]_[platform].csv``。
    3. 全部缺省：默认保存到 ``./csv/[主调文件名].csv``。

    **两种写入方式**：

    * ``log(**kwargs)`` — 快捷写入，参数名即列名，首次调用自动确定列结构。
      支持新列自动迁移（旧行用默认值填充）。
    * ``save_csv(dict_or_list)`` — 字典写入，需预先设置 ``x_columns`` / ``y_columns``。

    **统计与查重**：

    * ``count()`` — 按 x_columns 分组，对 y_columns 计算 mean/std。
    * ``clean_duplicates()`` — 去重。
    * ``is_sampled()`` — 查重（需开启 ``avoid_duplication``）。

    :param root: CSV 文件完整路径。若为 ``None`` 则从 ``args`` 或默认值推导。
    :param args: ``StandardTrainingPipeline.init()`` 返回的 args 对象，
                 从中读取 ``csv_dir``、``csv_tag``、``platform`` 等关键字。
    :param x_columns: 实验参数列名（用于 groupby）。若为 ``None``，则在 ``count`` /
                      ``clean_duplicates`` 时自动从数据中推断。
    :param y_columns: 实验结果列名（用于计算 mean、std）。若为 ``None``，则在 ``count``
                      时自动从数据中推断。
    :param str_columns: 需要按字符串处理的列名（解决类型匹配问题）。
    :param avoid_duplication: 是否开启查重防重复训练。
    :param platforms: 多平台后缀列表（如 ``['6001', '6002']``），用于自动合并多机数据。
    """

    def __init__(
        self,
        root: Optional[str] = None,
        args: Any = None,
        x_columns: Optional[Union[str, List[str]]] = None,
        y_columns: Optional[Union[str, List[str]]] = None,
        str_columns: Optional[Union[str, List[str]]] = None,
        avoid_duplication: bool = False,
        platforms: Optional[List[str]] = None,
    ):
        self.root = self._resolve_root(root, args)
        self.x_columns = self._parse_columns(x_columns)
        self.y_columns = self._parse_columns(y_columns)
        self.columns = self.x_columns + self.y_columns
        self.str_columns = set(self._parse_columns(str_columns))

        self.avoid_duplication = avoid_duplication
        self.platforms = platforms if platforms is not None else []
        self.df: Optional[pd.DataFrame] = None  # 内部维护的 DataFrame，用于查重
        self._header_written: bool = True  # 是否需要写入头行（True=需要写，False=已存在无需写）
        self._columns_checked: bool = False  # 数据列名校验是否已完成（True=已通过，后续跳过）
        self._defaults: Dict[str, Any] = {}  # 新增列的默认值

        # 确保保存目录存在
        os.makedirs(os.path.dirname(self.root) or ".", exist_ok=True)

        # 如果 CSV 文件已存在，验证头行列名与 x+y 列名是否一致
        self._validate_header()

        # 如果开启了防重复或指定了多平台，则加载历史记录
        if self.avoid_duplication or self.platforms:
            self._load_records()

    # ------------------------------------------------------------------
    #  路径推导
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_root(root: Optional[str], args: Any) -> str:
        """推导 CSV 文件路径。

        优先级：``root`` 参数 > ``args`` 中的关键字 > 默认值 ``./csv/``。
        """
        if root is not None:
            return root

        # 尝试从 args 中读取关键字
        csv_dir = None
        csv_tag = None
        platform = None

        if args is not None:
            csv_dir = getattr(args, "csv_dir", None)
            csv_tag = getattr(args, "csv_tag", None)
            platform = getattr(args, "platform", None)

        # 获取主调文件名
        try:
            from .utils import get_caller_filename
        except ImportError:
            from dlkit.utils import get_caller_filename
        caller_name = get_caller_filename(stem=True)

        # 拼接文件名: [主调文件名]_[tag]_[platform].csv
        parts = [caller_name]
        if csv_tag:
            parts.append(csv_tag)
        if platform:
            parts.append(platform)
        filename = "_".join(parts) + ".csv"

        # 确定保存目录
        save_dir = Path(csv_dir) if csv_dir else Path.cwd() / "csv"

        return str(save_dir / filename)

    # ------------------------------------------------------------------
    #  头行验证
    # ------------------------------------------------------------------
    def _validate_header(self):
        """检查已存在的 CSV 文件头行，并与参数列名比较。

        * 若文件存在且头行存在：与 ``self.columns`` 比对，不一致则报错；
          同时设 ``_header_written = False``（表示头行已存在，无需再写）。
        * 若文件不存在或无头行：``_header_written`` 保持默认 ``True``（表示需要写头行）。
        """
        if not os.path.exists(self.root):
            return

        # 读取头行（仅第一行，不加载全部数据）
        try:
            with open(self.root, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
        except Exception:
            return

        if not header:
            return  # 文件为空或无头行，保持 _header_written = True

        file_cols = [c.strip() for c in header]

        # 头行存在，标记为无需再写头行
        self._header_written = False

        # 若用户指定了列名，则进行比对
        if self.columns:
            expected_cols = list(self.columns)
            if set(file_cols) != set(expected_cols):
                missing = sorted(set(expected_cols) - set(file_cols))
                extra = sorted(set(file_cols) - set(expected_cols))
                raise ValueError(
                    f"CSV 文件头行列名与指定的 x_columns + y_columns 不一致。\n"
                    f"  文件: {self.root}\n"
                    f"  文件头行列名: {file_cols}\n"
                    f"  指定列名(x+y): {expected_cols}\n"
                    f"  缺失的列(指定有但文件无): {missing}\n"
                    f"  多余的列(文件有但指定无): {extra}"
                )
        else:
            # 用户未指定列名，则从文件头行继承
            self.columns = file_cols
            self.x_columns = self.x_columns or []
            self.y_columns = self.y_columns or []

    # ------------------------------------------------------------------
    #  列名解析
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_columns(cols: Union[str, List[str], None]) -> List[str]:
        """解析并规范化传入的列参数。

        支持逗号分隔的字符串（如 ``'margin, mse, lr'``）或列表（如 ``['margin', 'mse', 'lr']``）。
        """
        if cols is None:
            res: List[str] = []
        elif isinstance(cols, str):
            res = [c.strip() for c in cols.split(",") if c.strip()]
        elif isinstance(cols, (list, tuple)):
            res = [str(c).strip() for c in cols if str(c).strip()]
        else:
            raise ValueError(
                f"不支持的列格式: {type(cols)}。请传入以逗号分隔的字符串或列表。"
            )

        # 重复时报错
        seen: set[str] = set()
        for col in res:
            if col in seen:
                raise ValueError(f"重复列名: {col}。")
            seen.add(col)

        return res

    # ------------------------------------------------------------------
    #  自动推断 x / y 列
    # ------------------------------------------------------------------
    @staticmethod
    def _infer_columns(df: pd.DataFrame) -> tuple:
        """从 DataFrame 中自动推断 x_columns 和 y_columns。

        推断规则（按优先级从高到低）：

        1. **指标名匹配（最高优先级）**：列名匹配常见实验指标关键词
           （``acc``、``loss``、``error``、``auc``、``f1`` 等），
           **无论唯一值多少**，一律归为 y_columns。
           ``run_id`` 也归为 y，统计时得到 ``run_id_mean`` 无实际意义但不影响结果。

        2. **非数值列**：字符串、布尔等非数值类型的列归为 x_columns（离散参数）。

        3. **参数名匹配**：列名匹配常见超参数关键词
           （``lr``、``batch``、``seed``、``margin`` 等）的列归为 x_columns。

        4. **数值列阈值判定**：剩余的数值列中，唯一值数量较少
           （``<= max(10, len(df) * 0.3)``）的归为 x_columns（离散参数），
           唯一值较多的归为 y_columns（连续结果）。

        :param df: 已加载的 DataFrame。
        :return: ``(x_columns, y_columns)``。
        """
        if df is None or df.empty:
            return [], []

        n = len(df)
        all_cols = list(df.columns)

        # 常见实验指标名关键词（不区分大小写），匹配到的列优先归为 y
        metric_keywords = (
            "acc", "accuracy", "err", "error", "loss", "cost",
            "auc", "auroc", "ap", "map", "f1", "precision", "recall",
            "score", "metric", "top1", "top5", "topk",
            "val_acc", "test_acc", "train_acc",
            "bleu", "rouge", "perplexity", "ppl",
            "mse", "mae", "rmse", "nll", "ce", "cross_entropy",
            "ks", "ndcg", "mrr", "hit", "coverage",
            "tau", "kendall", "spearman",
            "run_id",
        )

        # 常见超参数名关键词（不区分大小写），匹配到的列归为 x
        param_keywords = (
            "lr", "learning_rate", "batch", "seed", "epoch", "margin",
            "weight_decay", "momentum", "dropout", "optimizer", "schedule",
            "warmup", "decay", "ratio", "alpha", "beta", "gamma", "lambda",
            "temperature", "smooth", "label", "mixup", "cutout", "augment",
        )

        x_cols: list = []
        y_cols: list = []

        for col in all_cols:
            col_lower = col.lower()
            series = df[col]
            n_unique = series.nunique(dropna=True)

            # 优先级 1: 指标名匹配 -> y（无论唯一值多少）
            if any(kw in col_lower for kw in metric_keywords):
                y_cols.append(col)
                continue

            # 优先级 2: 非数值列 -> x
            is_numeric = pd.api.types.is_numeric_dtype(series)
            if not is_numeric:
                x_cols.append(col)
                continue

            # 优先级 3: 参数名匹配 -> x
            if any(kw in col_lower for kw in param_keywords):
                x_cols.append(col)
                continue

            # 优先级 4: 数值列阈值判定
            threshold = max(10, int(n * 0.3))
            if n_unique <= threshold:
                x_cols.append(col)
            else:
                y_cols.append(col)

        return x_cols, y_cols

    # ------------------------------------------------------------------
    #  历史记录加载
    # ------------------------------------------------------------------
    def _load_records(self):
        """合并多个平台的 CSV 文件，或加载基础 CSV 文件。"""
        # 解析文件前缀：去掉 .csv，再去掉末尾的 _数字 后缀（平台标识）
        stem = self.root.rsplit(".", 1)[0]
        parts = stem.rsplit("_", 1)
        if len(parts) > 1 and parts[1].isdigit():
            prefix = parts[0]
        else:
            prefix = stem

        paths = (
            [f"{prefix}_{platform}.csv" for platform in self.platforms]
            if self.platforms
            else []
        )
        paths.append(self.root)
        paths = list(set(paths))
        dfs = []

        for path in paths:
            if os.path.exists(path):
                dfs.append(pd.read_csv(path))

        if dfs:
            self.df = pd.concat(dfs, axis=0, ignore_index=True)
            for col in self.str_columns:
                if col in self.df.columns:
                    self.df[col] = self.df[col].astype(str)
            print(f"[Tracker] Loaded existing records from base file: [{self.root}]")
        else:
            self.df = pd.DataFrame(columns=self.columns)
            print("[Tracker] No existing records found. Starting fresh.")

    # ------------------------------------------------------------------
    #  查重
    # ------------------------------------------------------------------
    def is_sampled(self, **kwargs) -> bool:
        """通用查重方法：传入需要匹配的键值对。

        keys must be among ``self.x_columns``。

        例如::

            tracker.is_sampled(mse=0.5, mse_ratio=0.3, margin=0.1, run_id=1)
        """
        if not self.avoid_duplication or self.df is None or self.df.empty:
            return False

        condition = pd.Series(True, index=self.df.index)
        for key, value in kwargs.items():
            condition &= self.df[key] == value

        return bool(condition.any())

    # ------------------------------------------------------------------
    #  保存
    # ------------------------------------------------------------------
    def save_csv(self, obj: Union[List[Dict], Dict], mode: str = "a"):
        """将实验结果写入 CSV 文件。

        需预先设置 ``x_columns`` / ``y_columns``。若出现新列则自动迁移。

        :param obj: 单条字典或多条字典列表。
        :param mode: 写入模式，``'a'`` 追加（默认），``'w'`` 覆写。
        """
        if not obj:
            return

        # 统一为列表
        if isinstance(obj, dict):
            obj = [obj]

        # 确定列名：优先使用已设置的，否则从数据 key 推导（保持顺序）
        if not self.columns:
            self.columns = list(obj[0].keys())

        # 检测新列并迁移（检查所有行）
        all_keys = set()
        for row in obj:
            all_keys.update(row.keys())
        new_cols = [k for k in all_keys if k not in self.columns]
        if new_cols:
            self._migrate_with_new_columns(obj[0], new_cols)

        # 校验列名一致性（仅首次检查）
        if not self._columns_checked:
            data_keys = set()
            for row in obj:
                data_keys.update(row.keys())
            col_set = set(self.columns)
            missing_keys = col_set - data_keys
            if missing_keys:
                raise ValueError(
                    f"写入数据缺少必要的 key（列名有但数据无）。\n"
                    f"  列名: {sorted(col_set)}\n"
                    f"  缺失的 key: {sorted(missing_keys)}"
                )
            self._columns_checked = True

        self._write_rows(obj, mode)

    # ------------------------------------------------------------------
    #  快捷写入（关键字参数自动映射为列名和值）
    # ------------------------------------------------------------------
    def set_defaults(self, **kwargs):
        """设置新增列的默认值。

        当后续 ``log()`` 调用中出现当前 CSV 中不存在的列时，
        旧数据行中该列将使用此处设置的默认值填充。
        未设置默认值的列默认为空字符串 ``''``。

        用法::

            tracker = CSVExpTracker(root='./csv/results.csv')
            # 已有列: lr, acc
            tracker.log(lr=0.01, acc=0.9)

            # 设置新列默认值
            tracker.set_defaults(loss=0.0, batch_size=32)

            # 后续 log 出现新列时自动迁移
            tracker.log(lr=0.02, acc=0.91, loss=0.05, batch_size=64)
            # 旧行中 loss 填 0.0，batch_size 填 32

        :param kwargs: 列名=默认值 的键值对。
        """
        self._defaults.update(kwargs)

    def log(self, **kwargs):
        """快捷写入：将关键字参数名作为列名、参数值作为值写入 CSV。

        无需预先设置 ``x_columns`` / ``y_columns``，首次调用时自动按参数顺序确定列名。

        当出现新列（当前 CSV 中不存在的列）时，自动执行迁移：
        读取旧数据、用 ``set_defaults()`` 设置的默认值（未设置则为空字符串）
        填充旧行中的新列、重写整个文件、然后追加新数据行。

        用法::

            tracker = CSVExpTracker(root='./csv/results.csv')
            tracker.log(lr=0.01, batch_size=32, acc=0.925, loss=0.05)

        首次调用时写入头行 ``lr,batch_size,acc,loss`` 和数据行。
        后续调用只需传入相同的关键字参数即可追加数据行。

        :param kwargs: 列名=值 的键值对，顺序决定 CSV 列顺序。
        """
        if not kwargs:
            return

        # 首次调用：按参数顺序确定列名
        if not self.columns:
            self.columns = list(kwargs.keys())

        # 检测新列
        new_cols = [k for k in kwargs if k not in self.columns]
        if new_cols:
            # 将新列插入到 kwargs 指定的位置
            self._migrate_with_new_columns(kwargs, new_cols)

        # 校验列名一致性（每次检查，set 比较开销极小）
        col_set = set(self.columns)
        kw_set = set(kwargs.keys())
        missing_keys = col_set - kw_set
        if missing_keys:
            raise ValueError(
                f"log() 参数缺少必要的 key（列名有但参数无）。\n"
                f"  列名: {self.columns}\n"
                f"  缺失的 key: {sorted(missing_keys)}"
            )
        self._columns_checked = True

        self._write_rows([kwargs], "a")

    # ------------------------------------------------------------------
    #  列迁移
    # ------------------------------------------------------------------
    def _migrate_with_new_columns(self, row_dict: Dict[str, Any], new_cols: List[str]):
        """将新列插入到现有 CSV 中，旧行用默认值填充。

        步骤：
        1. 读取旧 CSV 全部数据
        2. 按 row_dict 的顺序重建列名列表（新列插入到指定位置）
        3. 旧行中缺失的新列用 ``_defaults`` 或空字符串填充
        4. 重写整个文件（头行 + 旧数据行）
        5. 更新标志位

        :param row_dict: 当前写入行的完整键值对，决定列顺序。
        :param new_cols: 新列名列表。
        """
        # 1. 读取旧数据
        old_rows: List[Dict] = []
        if os.path.exists(self.root):
            try:
                with open(self.root, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    old_rows = list(reader)
            except Exception:
                old_rows = []

        # 2. 按 row_dict 顺序重建列名（新列插入到指定位置）
        self.columns = list(row_dict.keys())

        # 3. 旧行中缺失的新列用默认值填充
        for row in old_rows:
            for col in new_cols:
                row[col] = self._defaults.get(col, "")

        # 4. 重写整个文件
        with open(self.root, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writeheader()
            if old_rows:
                writer.writerows(old_rows)

        # 5. 更新标志位
        self._header_written = False
        self._columns_checked = False  # 列结构变了，下次需要重新校验
        print(f"[Tracker] 检测到新列 {new_cols}，已迁移旧数据并重写文件。")

    # ------------------------------------------------------------------
    #  内部写入
    # ------------------------------------------------------------------
    def _write_rows(self, rows: List[Dict], mode: str):
        """内部方法：将字典列表写入 CSV 文件。

        :param rows: 字典列表。
        :param mode: 写入模式，``'a'`` 或 ``'w'``。
        """
        write_mode = "a" if mode == "a" else "w"

        # 判断是否需要写入头行
        need_header = self._header_written or write_mode == "w"

        with open(self.root, write_mode, encoding="utf-8", newline="") as f:
            # extrasaction='raise'（默认）：多余 key 自动报错 ValueError
            writer = csv.DictWriter(f, fieldnames=self.columns)
            if need_header:
                writer.writeheader()
            writer.writerows(rows)

        # 更新标志位：头行已写入，后续追加不再写
        self._header_written = False

    # ------------------------------------------------------------------
    #  去重
    # ------------------------------------------------------------------
    def clean_duplicates(self, keep: str = "last", overwrite: bool = False) -> Optional[pd.DataFrame]:
        """清理 CSV 文件中的重复记录。

        若 ``x_columns`` 未指定，则自动从数据中推断（不修改实例状态）。

        :param keep: 保留哪一条重复记录（``'last'`` 或 ``'first'``）。
        :param overwrite: 是否将去重后结果覆盖原文件。
        :return: 去重后的 DataFrame。
        """
        if not os.path.exists(self.root):
            return None

        df = pd.read_csv(self.root)
        if df.empty:
            return df

        # 推断列名（使用局部变量，不修改实例状态）
        x_cols = list(self.x_columns) if self.x_columns else None
        if not x_cols:
            x_cols, _ = self._infer_columns(df)
            if x_cols:
                print(f"[Tracker] 自动推断 x_columns: {x_cols}")

        # 去重：x_columns + run_id（避免误删不同种子的行）
        subset = x_cols + ["run_id"] if "run_id" in df.columns else x_cols
        if not subset:
            print("[Tracker] 无法确定去重依据列，跳过去重。")
            return df

        before_len = len(df)
        df.drop_duplicates(subset=subset, keep=keep, inplace=True)
        after_len = len(df)

        if before_len > after_len:
            print(
                f"[Tracker] 清理了 {before_len - after_len} 条重复记录 "
                f"(依据列: {subset})。"
            )
            if overwrite:
                df.to_csv(self.root, index=False)
                if self.avoid_duplication:
                    self.df = df
        else:
            print("[Tracker] 数据干净，未发现重复记录。")

        return df

    # ------------------------------------------------------------------
    #  统计
    # ------------------------------------------------------------------
    def count(self):
        """对保存的结果进行聚合计算（Mean、Std）及分组计数。

        若 ``x_columns`` 或 ``y_columns`` 未指定，则自动从数据中推断
        （不修改实例状态）。

        统计结果保存至 ``<root>_statistics.csv``。
        """
        if not os.path.exists(self.root):
            print(f"[Tracker] CSV file {self.root} does not exist. Cannot count.")
            return

        df = self.clean_duplicates(keep="last", overwrite=False)
        if df is None:
            return
        df = df.dropna()
        if df.empty:
            print("[Tracker] Dataframe is empty. No statistics to compute.")
            return

        # 推断列名（使用局部变量，不修改实例状态）
        x_cols = list(self.x_columns) if self.x_columns else None
        y_cols = list(self.y_columns) if self.y_columns else None

        if not x_cols or not y_cols:
            inferred_x, inferred_y = self._infer_columns(df)
            if not x_cols:
                x_cols = inferred_x
                print(f"[Tracker] 自动推断 x_columns: {x_cols}")
            if not y_cols:
                y_cols = inferred_y
                print(f"[Tracker] 自动推断 y_columns: {y_cols}")

        if not y_cols:
            print("[Tracker] 无法推断 y_columns，无法计算统计量。")
            return
        if not x_cols:
            print("[Tracker] 无法推断 x_columns，无法分组。")
            return

        group = df.groupby(by=x_cols)

        valid_y_cols = [col for col in y_cols if col in df.columns]
        agg_dict = {col: ["mean", "std"] for col in valid_y_cols}
        des = group.agg(agg_dict)

        group_counts = group.size()
        des.columns = [f"{col}_{stat}" for col, stat in des.columns]
        des["total_runs"] = group_counts

        save_path = self.root.rsplit(".", 1)[0] + "_statistics.csv"
        des.to_csv(save_path)
        print(f"[Tracker] 统计数据已聚合保存至: {save_path}")


# ======================================================================
#  测试用例
# ======================================================================
if __name__ == "__main__":
    import shutil
    import tempfile

    _pass = 0
    _fail = 0

    def check(cond, msg):
        global _pass, _fail
        if cond:
            _pass += 1
            print(f"  ✓ {msg}")
        else:
            _fail += 1
            print(f"  ✗ FAIL: {msg}")

    def section(title):
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")

    tmpdir = tempfile.mkdtemp(prefix="tracker_test_")

    try:
        # ==============================================================
        section("1. _parse_columns")
        # ==============================================================
        check(CSVExpTracker._parse_columns(None) == [], "None -> []")
        check(CSVExpTracker._parse_columns("a, b, c") == ["a", "b", "c"], "str -> list")
        check(CSVExpTracker._parse_columns(["x", "y"]) == ["x", "y"], "list -> list")
        try:
            CSVExpTracker._parse_columns("a, b, a")
            check(False, "str with dup should raise")
        except ValueError:
            check(True, "str with dup raises ValueError")
        try:
            CSVExpTracker._parse_columns(["a", "a"])
            check(False, "dup in list should raise")
        except ValueError:
            check(True, "dup in list raises ValueError")
        try:
            CSVExpTracker._parse_columns(123)
            check(False, "int should raise")
        except ValueError:
            check(True, "int raises ValueError")

        # ==============================================================
        section("2. _infer_columns — 指标名匹配")
        # ==============================================================
        df = pd.DataFrame({
            "lr": [0.01, 0.02, 0.03],
            "acc": [0.90, 0.91, 0.92],
            "loss": [0.10, 0.09, 0.08],
            "test_error": [0.05, 0.04, 0.03],
            "f1_score": [0.80, 0.81, 0.82],
            "run_id": [0, 1, 2],
        })
        x, y = CSVExpTracker._infer_columns(df)
        check("lr" in x, f"lr in x (got x={x})")
        check("acc" in y, f"acc in y (got y={y})")
        check("loss" in y, f"loss in y")
        check("test_error" in y, f"test_error in y")
        check("f1_score" in y, f"f1_score in y")
        check("run_id" in y, f"run_id in y")

        # ==============================================================
        section("3. _infer_columns — 非数值列归 x")
        # ==============================================================
        df2 = pd.DataFrame({
            "optimizer": ["adam", "sgd", "adamw"],
            "model": ["resnet", "vgg", "densenet"],
            "acc": [0.90, 0.91, 0.92],
        })
        x2, y2 = CSVExpTracker._infer_columns(df2)
        check("optimizer" in x2, f"optimizer in x (got x={x2})")
        check("model" in x2, f"model in x")
        check("acc" in y2, f"acc in y (got y={y2})")

        # ==============================================================
        section("4. _infer_columns — 大 n 但 y 取值少（指标名优先）")
        # ==============================================================
        n = 200
        df3 = pd.DataFrame({
            "lr": [0.01] * 100 + [0.02] * 100,
            "acc": [0.90, 0.91, 0.92, 0.93] * 50,
            "loss": [0.10, 0.09, 0.08, 0.07] * 50,
        })
        x3, y3 = CSVExpTracker._infer_columns(df3)
        check("lr" in x3, f"lr in x")
        check("acc" in y3, f"acc in y (even with few unique values)")
        check("loss" in y3, f"loss in y")

        # ==============================================================
        section("5. _infer_columns — 空 DataFrame")
        # ==============================================================
        x5, y5 = CSVExpTracker._infer_columns(pd.DataFrame())
        check(x5 == [] and y5 == [], "empty df -> ([], [])")
        x5b, y5b = CSVExpTracker._infer_columns(None)
        check(x5b == [] and y5b == [], "None -> ([], [])")

        # ==============================================================
        section("6. log — 基本写入")
        # ==============================================================
        p6 = os.path.join(tmpdir, "t6.csv")
        t6 = CSVExpTracker(root=p6)
        t6.log(lr=0.01, batch_size=32, acc=0.925, loss=0.05)
        with open(p6) as f:
            lines = f.readlines()
        check(lines[0].strip() == "lr,batch_size,acc,loss", f"header: {lines[0].strip()}")
        check(lines[1].strip() == "0.01,32,0.925,0.05", f"data: {lines[1].strip()}")
        check(t6.columns == ["lr", "batch_size", "acc", "loss"], "columns order preserved")

        # ==============================================================
        section("7. log — 追加不重复写头行")
        # ==============================================================
        t6.log(lr=0.02, batch_size=64, acc=0.93, loss=0.04)
        with open(p6) as f:
            lines = f.readlines()
        check(len(lines) == 3, f"3 lines (1 header + 2 data), got {len(lines)}")

        # ==============================================================
        section("8. log — 连续大量追加")
        # ==============================================================
        for i in range(100):
            t6.log(lr=0.01 * (i + 3), batch_size=32, acc=0.9 + 0.001 * i, loss=0.05 - 0.0001 * i)
        with open(p6) as f:
            lines = f.readlines()
        check(len(lines) == 103, f"103 lines, got {len(lines)}")

        # ==============================================================
        section("9. log — 缺失 key 报错")
        # ==============================================================
        p9 = os.path.join(tmpdir, "t9.csv")
        t9 = CSVExpTracker(root=p9)
        t9.log(lr=0.01, acc=0.9, loss=0.05)
        try:
            t9.log(lr=0.02, acc=0.91)  # 缺 loss
            check(False, "should raise")
        except ValueError as e:
            check("loss" in str(e), f"missing key error: {e}")

        # ==============================================================
        section("10. log — 多余 key 自动迁移为新列")
        # ==============================================================
        # unknown 被检测为新列并自动迁移，不会报错
        t9.log(lr=0.03, acc=0.92, loss=0.04, unknown=123)
        with open(p9) as f:
            content = f.read()
        lines = content.strip().split("\n")
        check("unknown" in lines[0], f"unknown added as new column: {lines[0]}")
        check("123" in lines[-1], f"unknown value written: {lines[-1]}")

        # ==============================================================
        section("11. log — 空 kwargs 不报错")
        # ==============================================================
        t9.log()  # 不应报错
        check(True, "log() with no args is no-op")

        # ==============================================================
        section("12. set_defaults + log 新列迁移")
        # ==============================================================
        p12 = os.path.join(tmpdir, "t12.csv")
        t12 = CSVExpTracker(root=p12)
        t12.log(lr=0.01, acc=0.9)
        t12.log(lr=0.02, acc=0.91)
        t12.set_defaults(loss=0.0, batch_size=32)
        t12.log(lr=0.03, acc=0.92, loss=0.05, batch_size=64)
        with open(p12) as f:
            content = f.read()
        lines = content.strip().split("\n")
        check(lines[0] == "lr,acc,loss,batch_size", f"header: {lines[0]}")
        check("0.01,0.9,0.0,32" in lines[1], f"old row 1: {lines[1]}")
        check("0.02,0.91,0.0,32" in lines[2], f"old row 2: {lines[2]}")
        check("0.03,0.92,0.05,64" in lines[3], f"new row: {lines[3]}")

        # ==============================================================
        section("13. 新列迁移 — 无默认值用空字符串")
        # ==============================================================
        p13 = os.path.join(tmpdir, "t13.csv")
        t13 = CSVExpTracker(root=p13)
        t13.log(lr=0.01, acc=0.9)
        t13.log(lr=0.02, acc=0.91, loss=0.05)
        with open(p13) as f:
            content = f.read()
        lines = content.strip().split("\n")
        check("0.01,0.9," in lines[1], f"old row empty loss: {lines[1]}")

        # ==============================================================
        section("14. 新列迁移 — 插入到中间位置")
        # ==============================================================
        p14 = os.path.join(tmpdir, "t14.csv")
        t14 = CSVExpTracker(root=p14)
        t14.log(lr=0.01, acc=0.9)
        t14.set_defaults(loss=0.0)
        t14.log(lr=0.02, loss=0.05, acc=0.91)  # loss 在中间
        with open(p14) as f:
            content = f.read()
        lines = content.strip().split("\n")
        check(lines[0] == "lr,loss,acc", f"header: {lines[0]}")
        check("0.01,0.0,0.9" in lines[1], f"old row: {lines[1]}")

        # ==============================================================
        section("15. 多次新列迁移")
        # ==============================================================
        p15 = os.path.join(tmpdir, "t15.csv")
        t15 = CSVExpTracker(root=p15)
        t15.log(lr=0.01, acc=0.9)
        t15.set_defaults(loss=0.0)
        t15.log(lr=0.02, acc=0.91, loss=0.05)
        t15.set_defaults(batch_size=32)
        t15.log(lr=0.03, acc=0.92, loss=0.04, batch_size=64)
        with open(p15) as f:
            content = f.read()
        lines = content.strip().split("\n")
        check(len(lines) == 4, f"4 lines, got {len(lines)}")
        check("0.01,0.9,0.0,32" in lines[1], f"row1: {lines[1]}")
        check("0.02,0.91,0.05,32" in lines[2], f"row2: {lines[2]}")
        check("0.03,0.92,0.04,64" in lines[3], f"row3: {lines[3]}")

        # ==============================================================
        section("16. save_csv — 基本写入")
        # ==============================================================
        p16 = os.path.join(tmpdir, "t16.csv")
        t16 = CSVExpTracker(root=p16, x_columns=["lr"], y_columns=["acc", "loss"])
        t16.save_csv({"lr": 0.01, "acc": 0.9, "loss": 0.05})
        with open(p16) as f:
            lines = f.readlines()
        check(lines[0].strip() == "lr,acc,loss", f"header: {lines[0].strip()}")
        check(lines[1].strip() == "0.01,0.9,0.05", f"data: {lines[1].strip()}")

        # ==============================================================
        section("17. save_csv — 多行写入")
        # ==============================================================
        t16.save_csv([
            {"lr": 0.02, "acc": 0.91, "loss": 0.04},
            {"lr": 0.03, "acc": 0.92, "loss": 0.03},
        ])
        with open(p16) as f:
            lines = f.readlines()
        check(len(lines) == 4, f"4 lines, got {len(lines)}")

        # ==============================================================
        section("18. save_csv — 覆写模式")
        # ==============================================================
        t16.save_csv({"lr": 0.001, "acc": 0.99, "loss": 0.01}, mode="w")
        with open(p16) as f:
            lines = f.readlines()
        check(len(lines) == 2, f"2 lines after overwrite, got {len(lines)}")
        check(lines[0].strip() == "lr,acc,loss", "header present")

        # ==============================================================
        section("19. save_csv — 新列迁移")
        # ==============================================================
        p19 = os.path.join(tmpdir, "t19.csv")
        t19 = CSVExpTracker(root=p19, x_columns=["lr"], y_columns=["acc"])
        t19.save_csv({"lr": 0.01, "acc": 0.9})
        t19.save_csv({"lr": 0.02, "acc": 0.91, "loss": 0.05})
        with open(p19) as f:
            content = f.read()
        lines = content.strip().split("\n")
        check(lines[0] == "lr,acc,loss", f"header: {lines[0]}")
        check("0.01,0.9," in lines[1], f"old row: {lines[1]}")

        # ==============================================================
        section("20. save_csv — 缺失 key 报错")
        # ==============================================================
        p20 = os.path.join(tmpdir, "t20.csv")
        t20 = CSVExpTracker(root=p20, x_columns=["lr"], y_columns=["acc", "loss"])
        try:
            t20.save_csv({"lr": 0.03})  # 缺 acc, loss
            check(False, "should raise")
        except ValueError as e:
            check("acc" in str(e) or "loss" in str(e), f"missing key: {e}")

        # ==============================================================
        section("21. save_csv — 空 obj 不报错")
        # ==============================================================
        t19.save_csv([])
        t19.save_csv(None)
        check(True, "empty obj is no-op")

        # ==============================================================
        section("22. _validate_header — 文件不存在")
        # ==============================================================
        p22 = os.path.join(tmpdir, "t22.csv")
        t22 = CSVExpTracker(root=p22, x_columns=["lr"], y_columns=["acc"])
        check(t22._header_written == True, "new file: _header_written=True")

        # ==============================================================
        section("23. _validate_header — 文件存在且头行一致")
        # ==============================================================
        t22.log(lr=0.01, acc=0.9)
        t23 = CSVExpTracker(root=p22, x_columns=["lr"], y_columns=["acc"])
        check(t23._header_written == False, "existing file with header: _header_written=False")

        # ==============================================================
        section("24. _validate_header — 头行不一致报错")
        # ==============================================================
        try:
            CSVExpTracker(root=p22, x_columns=["lr", "batch_size"], y_columns=["acc"])
            check(False, "should raise")
        except ValueError as e:
            check("不一致" in str(e) and "batch_size" in str(e), f"mismatch error: {e}")

        # ==============================================================
        section("25. _validate_header — 未指定列名从文件继承")
        # ==============================================================
        t25 = CSVExpTracker(root=p22)
        check(t25.columns == ["lr", "acc"], f"inherited columns: {t25.columns}")

        # ==============================================================
        section("26. _validate_header — 空文件")
        # ==============================================================
        p26 = os.path.join(tmpdir, "t26.csv")
        open(p26, "w").close()
        t26 = CSVExpTracker(root=p26, x_columns=["lr"], y_columns=["acc"])
        check(t26._header_written == True, "empty file: _header_written=True")
        t26.log(lr=0.01, acc=0.9)
        with open(p26) as f:
            lines = f.readlines()
        check(lines[0].strip() == "lr,acc", "header written for empty file")

        # ==============================================================
        section("27. clean_duplicates — 去重")
        # ==============================================================
        p27 = os.path.join(tmpdir, "t27.csv")
        t27 = CSVExpTracker(root=p27)
        t27.log(lr=0.01, acc=0.9, run_id=0)
        t27.log(lr=0.01, acc=0.9, run_id=0)  # 重复
        t27.log(lr=0.01, acc=0.91, run_id=1)
        t27.log(lr=0.02, acc=0.92, run_id=2)
        t28 = CSVExpTracker(root=p27)
        df = t28.clean_duplicates(overwrite=True)
        check(len(df) == 3, f"3 rows after dedup, got {len(df)}")
        with open(p27) as f:
            lines = f.readlines()
        check(len(lines) == 4, f"file has 4 lines (header+3), got {len(lines)}")

        # ==============================================================
        section("28. clean_duplicates — 不修改实例状态")
        # ==============================================================
        t29 = CSVExpTracker(root=p27)
        orig_x = list(t29.x_columns)
        orig_y = list(t29.y_columns)
        orig_cols = list(t29.columns)
        t29.clean_duplicates()
        check(t29.x_columns == orig_x, "x_columns unchanged")
        check(t29.y_columns == orig_y, "y_columns unchanged")
        check(t29.columns == orig_cols, "columns unchanged")

        # ==============================================================
        section("29. clean_duplicates — 文件不存在")
        # ==============================================================
        p29 = os.path.join(tmpdir, "t29_nonexist.csv")
        t29b = CSVExpTracker(root=p29)
        check(t29b.clean_duplicates() is None, "nonexistent file -> None")

        # ==============================================================
        section("30. count — 基本统计")
        # ==============================================================
        p30 = os.path.join(tmpdir, "t30.csv")
        t30 = CSVExpTracker(root=p30)
        for run_id in range(3):
            for lr in [0.01, 0.02]:
                t30.log(run_id=run_id, lr=lr, acc=0.9 + 0.01 * run_id + lr, loss=0.1 - 0.01 * run_id)
        t31 = CSVExpTracker(root=p30)
        t31.count()
        stat_path = p30.replace(".csv", "_statistics.csv")
        stat = pd.read_csv(stat_path)
        check("lr" in stat.columns, f"lr in stats: {stat.columns.tolist()}")
        check("acc_mean" in stat.columns, "acc_mean in stats")
        check("acc_std" in stat.columns, "acc_std in stats")
        check("loss_mean" in stat.columns, "loss_mean in stats")
        check("total_runs" in stat.columns, "total_runs in stats")
        check(stat["total_runs"].iloc[0] == 3, f"3 runs per lr group, got {stat['total_runs'].iloc[0]}")
        check(stat["total_runs"].iloc[1] == 3, f"3 runs per lr group, got {stat['total_runs'].iloc[1]}")

        # ==============================================================
        section("31. count — 不修改实例状态")
        # ==============================================================
        t32 = CSVExpTracker(root=p30)
        orig_x32 = list(t32.x_columns)
        orig_y32 = list(t32.y_columns)
        t32.count()
        check(t32.x_columns == orig_x32, "x_columns unchanged after count")
        check(t32.y_columns == orig_y32, "y_columns unchanged after count")

        # ==============================================================
        section("32. count — 文件不存在")
        # ==============================================================
        p32 = os.path.join(tmpdir, "t32_nonexist.csv")
        t32b = CSVExpTracker(root=p32)
        t32b.count()  # 不应报错
        check(True, "count on nonexistent file is no-op")

        # ==============================================================
        section("33. is_sampled — 查重")
        # ==============================================================
        p33 = os.path.join(tmpdir, "t33.csv")
        # 先写入数据
        t33_prep = CSVExpTracker(root=p33)
        t33_prep.log(lr=0.01, acc=0.9)
        t33_prep.log(lr=0.02, acc=0.91)
        # 重新打开，开启 avoid_duplication 加载历史
        t33 = CSVExpTracker(root=p33, x_columns=["lr"], y_columns=["acc"], avoid_duplication=True)
        check(t33.is_sampled(lr=0.01) == True, "lr=0.01 is sampled")
        check(t33.is_sampled(lr=0.03) == False, "lr=0.03 is not sampled")

        # ==============================================================
        section("34. is_sampled — 未开启 avoid_duplication")
        # ==============================================================
        t34 = CSVExpTracker(root=p33)
        check(t34.is_sampled(lr=0.01) == False, "always False without avoid_duplication")

        # ==============================================================
        section("35. _resolve_root — 直接指定 root")
        # ==============================================================
        r35 = CSVExpTracker._resolve_root("/path/to/file.csv", None)
        check(r35 == "/path/to/file.csv", f"direct root: {r35}")

        # ==============================================================
        section("36. _resolve_root — 从 args 推导")
        # ==============================================================
        from argparse import Namespace
        args36 = Namespace(csv_dir="./my_csv", csv_tag="search", platform="6001")
        r36 = CSVExpTracker._resolve_root(None, args36)
        check("my_csv" in r36, f"csv_dir in path: {r36}")
        check("search" in r36, f"csv_tag in path: {r36}")
        check("6001" in r36, f"platform in path: {r36}")
        check(r36.endswith(".csv"), f"ends with .csv: {r36}")

        # ==============================================================
        section("37. _resolve_root — 默认路径")
        # ==============================================================
        r37 = CSVExpTracker._resolve_root(None, None)
        check("/csv/" in r37 or r37.startswith("csv"), f"default csv dir: {r37}")
        check(r37.endswith(".csv"), f"ends with .csv: {r37}")

        # ==============================================================
        section("38. _load_records — 多平台合并")
        # ==============================================================
        p38_base = os.path.join(tmpdir, "results")
        # 创建两个平台文件
        for plat in ["6001", "6002"]:
            pp = f"{p38_base}_{plat}.csv"
            t = CSVExpTracker(root=pp, x_columns=["lr"], y_columns=["acc"])
            t.save_csv({"lr": 0.01, "acc": 0.9})
        t38 = CSVExpTracker(
            root=f"{p38_base}_6001.csv",
            x_columns=["lr"], y_columns=["acc"],
            avoid_duplication=True, platforms=["6001", "6002"],
        )
        check(t38.df is not None and len(t38.df) == 2, f"merged 2 platform files, got {len(t38.df) if t38.df is not None else 0} rows")

        # ==============================================================
        section("39. _load_records — 前缀解析（多下划线）")
        # ==============================================================
        p39 = os.path.join(tmpdir, "my_search_results_6001.csv")
        t39 = CSVExpTracker(root=p39, platforms=["6001", "6002"], avoid_duplication=True)
        check(True, "multi-underscore prefix parse no error")

        # ==============================================================
        section("40. init — 自动创建目录")
        # ==============================================================
        deep = os.path.join(tmpdir, "a", "b", "c", "deep.csv")
        t40 = CSVExpTracker(root=deep, x_columns=["lr"], y_columns=["acc"])
        check(os.path.isdir(os.path.dirname(deep)), "deep dir created at init")
        t40.log(lr=0.01, acc=0.9)
        check(os.path.isfile(deep), "file written in deep dir")

        # ==============================================================
        section("41. log + save_csv 混用")
        # ==============================================================
        p41 = os.path.join(tmpdir, "t41.csv")
        t41 = CSVExpTracker(root=p41)
        t41.log(lr=0.01, acc=0.9)
        t41.save_csv({"lr": 0.02, "acc": 0.91})
        with open(p41) as f:
            lines = f.readlines()
        check(len(lines) == 3, f"3 lines (log + save_csv), got {len(lines)}")

        # ==============================================================
        section("42. 迁移后继续正常追加")
        # ==============================================================
        t15.log(lr=0.04, acc=0.93, loss=0.03, batch_size=128)
        with open(p15) as f:
            lines = f.readlines()
        check(len(lines) == 5, f"5 lines after migration+append, got {len(lines)}")

        # ==============================================================
        section("43. _header_written 标志位正确性")
        # ==============================================================
        p43 = os.path.join(tmpdir, "t43.csv")
        t43 = CSVExpTracker(root=p43)
        check(t43._header_written == True, "new: True")
        t43.log(lr=0.01, acc=0.9)
        check(t43._header_written == False, "after write: False")
        t43.log(lr=0.02, acc=0.91)
        check(t43._header_written == False, "after append: still False")
        t43.save_csv({"lr": 0.03, "acc": 0.92}, mode="w")
        check(t43._header_written == False, "after overwrite: False")

        # ==============================================================
        section("44. _columns_checked 标志位正确性")
        # ==============================================================
        p44 = os.path.join(tmpdir, "t44.csv")
        t44 = CSVExpTracker(root=p44)
        check(t44._columns_checked == False, "new: False")
        t44.log(lr=0.01, acc=0.9)
        check(t44._columns_checked == True, "after first log: True")
        t44.log(lr=0.02, acc=0.91)
        check(t44._columns_checked == True, "after second log: still True")

        # ==============================================================
        section("45. 迁移后 _columns_checked 重置")
        # ==============================================================
        t44.set_defaults(loss=0.0)
        t44.log(lr=0.03, acc=0.92, loss=0.05)
        check(t44._columns_checked == True, "after migration+log: True (re-checked)")

    finally:
        shutil.rmtree(tmpdir)

    # ================================================================
    print(f"\n{'='*60}")
    print(f"  结果: {_pass} 通过, {_fail} 失败 (共 {_pass + _fail} 项)")
    print(f"{'='*60}")
    if _fail > 0:
        raise SystemExit(1)
