"""
dlkit.argparse_utils
====================

支持多层级分组的命令行参数解析器 ``GroupArgparse``。

采用极简设计，默认根节点为顶层。支持按层级路径（如 ``'model/encoder'``）注册命令行参数，
并与基础配置（字典或 YAML）深度合并。解析后可输出嵌套 ``Namespace`` 或动态 ``Dataclass``。

清理说明：移除了废弃的 ``_export_dataclass_schema1`` 方法。
"""

from __future__ import annotations

import argparse
from dataclasses import make_dataclass
from typing import Any, Dict, List, Optional

import yaml

__all__ = ["GroupArgparse"]


class GroupArgparse:
    """支持多层级分组的命令行参数解析器。

    采用极简设计，默认根节点为顶层。支持按层级路径（如 ``'model/encoder'``）注册命令行参数，
    并与基础配置（字典或 YAML）深度合并。

    :param description: argparse 描述。
    :param base_config: 基础配置字典。
    :param yaml_path: YAML 配置文件路径。
    """

    def __init__(
        self,
        description: Optional[str] = None,
        base_config: Optional[Dict[str, Any]] = None,
        yaml_path: Optional[str] = None,
    ):
        self._parser = argparse.ArgumentParser(description=description)
        self._current_group: List[str] = []

        self.config: Dict[str, Any] = {}
        self.args: argparse.Namespace = argparse.Namespace()

        if yaml_path is not None:
            with open(yaml_path, "r", encoding="utf-8") as f:
                yaml_config = yaml.safe_load(f) or {}
                self.config = self._recursive_merge(yaml_config, self.config)

        if base_config is not None:
            self.config = self._recursive_merge(base_config, self.config)

    # ------------------------------------------------------------------
    #  分组管理
    # ------------------------------------------------------------------
    def set_cur_group(self, *args):
        """设置当前的分组路径。

        支持的调用方式::

            gparser.set_cur_group()                   # 恢复默认分组
            gparser.set_cur_group(None)               # 恢复默认分组
            gparser.set_cur_group('train')            # ['train']
            gparser.set_cur_group('model/encoder')    # ['model', 'encoder']
            gparser.set_cur_group('model', 'layer')   # ['model', 'layer']
            gparser.set_cur_group(['model', 'layer']) # ['model', 'layer']
        """
        if not args or (len(args) == 1 and args[0] is None):
            self._current_group = []
            return

        current: List[str] = []
        for arg in args:
            if isinstance(arg, str):
                current.extend(arg.split("/"))
            elif isinstance(arg, (list, tuple)):
                for v in arg:
                    current.extend(v.split("/"))
            else:
                raise TypeError(f"不支持的分组参数类型: {type(arg)}")

        self._current_group = current

    # ------------------------------------------------------------------
    #  注册参数
    # ------------------------------------------------------------------
    def add_argument(self, *args: str, **kwargs):
        """向当前分组注册命令行参数。

        支持同时传入多个短选项/长选项别名（例如: ``'-lr', '--learning-rate'``）。
        """
        # 1. 确定基础 dest
        dest = kwargs.get("dest")
        if not dest:
            long_opts = [a for a in args if a.startswith("--")]
            if long_opts:
                dest = long_opts[0].lstrip("-").replace("-", "_")
            else:
                short_opts = [a for a in args if a.startswith("-")]
                if short_opts:
                    dest = short_opts[0].lstrip("-").replace("-", "_")
                else:
                    dest = args[0]

        # 2. 拼接内部存放的完整路径 (full_dest)
        full_dest = "/".join(self._current_group + [dest])

        is_positional = not any(a.startswith("-") for a in args)

        if not is_positional:
            kwargs["dest"] = full_dest

        # 3. 拼接对外的命令行参数名 (CLI Flags)
        prefixed_args: List[str] = []
        for arg in args:
            if arg.startswith("-"):
                arg_wo_prefix = arg.lstrip("-")
                assert "/" not in arg_wo_prefix, f'命令行参数不能包含 "/": {arg}'
                full_cli_name = "/".join(self._current_group + [arg_wo_prefix])
                prefixed_args.append(f"--{full_cli_name}")
            else:
                assert "/" not in arg, f'命令行参数不能包含 "/": {arg}'
                prefixed_args.append(full_dest)

        if is_positional:
            kwargs.pop("dest", None)

        self._parser.add_argument(*prefixed_args, **kwargs)

    # ------------------------------------------------------------------
    #  解析
    # ------------------------------------------------------------------
    def parse_args(
        self,
        export_dataclass: bool = False,
        schema_path: str = "args_schema.py",
    ) -> Any:
        """解析命令行参数并与基础配置合并。"""
        parsed_args, _ = self._parser.parse_known_args()

        cmd_config: Dict[str, Any] = {}
        for arg_dest, value in vars(parsed_args).items():
            self._set_nested_value(cmd_config, arg_dest.split("/"), value)

        self.config = self._recursive_merge(cmd_config, self.config)
        self.args = self.nested_args(
            self.config, export_dataclass, schema_path
        )
        return self.args

    @staticmethod
    def nested_args(
        config: Dict[str, Any],
        export_dataclass: bool = False,
        schema_path: str = "args_schema.py",
        class_name: str = "GroupArgs",
    ) -> Any:
        """将 config 字典转换为嵌套 Namespace 或 Dataclass。"""
        if export_dataclass:
            GroupArgparse._export_dataclass_schema(config, schema_path)
            args = GroupArgparse._dict_to_dataclass(class_name, config)
        else:
            args = GroupArgparse._dict_to_nested_namespace(config)
        return args

    # ------------------------------------------------------------------
    #  字典 → Dataclass / Namespace
    # ------------------------------------------------------------------
    @staticmethod
    def _dict_to_dataclass(class_name: str, config_dict: Dict[str, Any]) -> Any:
        """在运行时将多层嵌套字典转化为原生的 Dataclass 实例。"""
        fields = []
        init_kwargs: Dict[str, Any] = {}

        for k, v in config_dict.items():
            if isinstance(v, dict):
                nested_class_name = (
                    "".join(word.capitalize() for word in k.split("_")) + "Config"
                )
                nested_instance = GroupArgparse._dict_to_dataclass(
                    nested_class_name, v
                )
                fields.append((k, type(nested_instance)))
                init_kwargs[k] = nested_instance
            else:
                val_type = type(v) if v is not None else Any
                fields.append((k, val_type))
                init_kwargs[k] = v

        dynamic_cls = make_dataclass(cls_name=class_name, fields=fields)
        return dynamic_cls(**init_kwargs)

    @staticmethod
    def _dict_to_nested_namespace(d: Any) -> Any:
        """递归地将字典转换为嵌套的 ``argparse.Namespace`` 对象。

        让字典支持点操作符: ``config['train']['lr']`` -> ``args.train.lr``
        """
        if isinstance(d, dict):
            return argparse.Namespace(
                **{
                    k: GroupArgparse._dict_to_nested_namespace(v)
                    for k, v in d.items()
                }
            )
        elif isinstance(d, list):
            return [GroupArgparse._dict_to_nested_namespace(v) for v in d]
        else:
            return d

    # ------------------------------------------------------------------
    #  导出 Dataclass Schema
    # ------------------------------------------------------------------
    @staticmethod
    def _export_dataclass_schema(
        config: Dict[str, Any], output_file: str = "config_schema.py"
    ):
        """读取当前 config，自动生成静态 Dataclass 定义文件供 IDE 提示使用。

        支持任意深度的嵌套配置。
        """
        lines = [
            "# ==================================================================",
            "# 由 GroupArgparse 自动生成的实验配置 Dataclass",
            "# (Auto-generated Config Dataclass)",
            "# 通过引用此文件可以使用具有代码补全提示的嵌套数据类，并构建实例对象",
            f"# from {output_file.split('.')[0]} import GroupArgs",
            "# args = GroupArgs()",
            "# 请勿直接修改此文件！(Do not modify directly)",
            "# ==================================================================",
            "from dataclasses import dataclass, field",
            "from typing import Any, Optional, List, Dict",
            "",
        ]

        generated_classes: Dict[str, bool] = {}

        def generate_class_from_dict(
            dict_obj: Dict[str, Any],
            class_name_suffix: str = "Config",
            parent_path: str = "",
        ) -> List[str]:
            """递归生成嵌套的 Dataclass 定义。"""
            subclasses: List[str] = []
            for key, value in dict_obj.items():
                if isinstance(value, dict):
                    child_path = (
                        f"{parent_path}/{key}" if parent_path else key
                    )
                    child_class_name = (
                        "".join(
                            part.capitalize()
                            for part in child_path.replace("/", "_").split("_")
                        )
                        + class_name_suffix
                    )
                    if child_class_name not in generated_classes:
                        child_lines = generate_class_from_dict(
                            value, class_name_suffix, child_path
                        )
                        subclasses.extend(child_lines)
                        generated_classes[child_class_name] = True

            if parent_path:
                current_class_name = (
                    "".join(
                        part.capitalize()
                        for part in parent_path.replace("/", "_").split("_")
                    )
                    + class_name_suffix
                )
            else:
                current_class_name = "GroupArgs"

            class_lines = [f"@dataclass", f"class {current_class_name}:"]

            for key, value in dict_obj.items():
                if isinstance(value, dict):
                    child_path = (
                        f"{parent_path}/{key}" if parent_path else key
                    )
                    child_class_name = (
                        "".join(
                            part.capitalize()
                            for part in child_path.replace("/", "_").split("_")
                        )
                        + class_name_suffix
                    )
                    class_lines.append(
                        f"    {key}: {child_class_name} = "
                        f"field(default_factory={child_class_name})"
                    )
                else:
                    class_lines.append(
                        GroupArgparse._generate_field_line(key, value)
                    )

            if len(class_lines) == 2:
                class_lines.append("    pass")

            class_lines.append("")
            return subclasses + class_lines

        all_class_lines = generate_class_from_dict(config, "Config", "")
        lines.extend(all_class_lines)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"静态配置类已成功导出至: {output_file}")

    @staticmethod
    def _generate_field_line(field_name: str, value: Any, indent: int = 4) -> str:
        """生成字段定义行。"""
        indent_str = " " * indent

        if value is None:
            return f"{indent_str}{field_name}: Any = None"
        elif isinstance(value, str):
            return f"{indent_str}{field_name}: str = '{value}'"
        elif isinstance(value, bool):
            return f"{indent_str}{field_name}: bool = {str(value)}"
        elif isinstance(value, int):
            return f"{indent_str}{field_name}: int = {value}"
        elif isinstance(value, float):
            return f"{indent_str}{field_name}: float = {value}"
        elif isinstance(value, list):
            return (
                f"{indent_str}{field_name}: List = "
                f"field(default_factory=lambda: {value})"
            )
        elif isinstance(value, dict):
            return (
                f"{indent_str}{field_name}: Dict = "
                f"field(default_factory=lambda: {value})"
            )
        else:
            return f"{indent_str}{field_name}: Any = {repr(value)}"

    # ------------------------------------------------------------------
    #  嵌套操作辅助
    # ------------------------------------------------------------------
    @staticmethod
    def _set_nested_value(
        config_group: Dict[str, Any], keys: List[str], value: Any
    ):
        """嵌套设置 config 的 key-value。"""
        if not keys:
            return
        current_level = config_group
        for key in keys[:-1]:
            current_level = current_level.setdefault(key, {})
        current_level[keys[-1]] = value

    @staticmethod
    def _recursive_merge(
        config_new: Dict[str, Any], config_old: Dict[str, Any]
    ) -> Dict[str, Any]:
        """将 config_new 覆盖合并到 config_old 中。"""
        for key, value in config_new.items():
            if isinstance(value, dict) and isinstance(config_old.get(key), dict):
                GroupArgparse._recursive_merge(value, config_old[key])
            else:
                config_old[key] = value
        return config_old

    def _flatten_config(self, config: Dict, result: Dict):
        """将嵌套 config 扁平化。"""
        for key, value in config.items():
            if isinstance(value, dict):
                self._flatten_config(value, result)
            else:
                result[key] = value
