# dlkit

> Deep Learning experiment assistant toolkit — 深度学习实验辅助工具包

`dlkit` 是一个用于深度学习实验的辅助工具包，提供从配置解析、设备分配、实验目录管理、日志记录到训练状态保存/加载的全流程辅助功能。

## 安装

```bash
pip install dlkit-zihao
```

或从源码安装：

```bash
cd dlkit
pip install -e .
```

## 功能模块

| 模块 | 主要类/函数 | 说明 |
|------|------------|------|
| `dlkit.pipeline` | `StandardTrainingPipeline` | 标准化实验流水线：配置解析、设备分配、种子设定、目录管理、日志、代码备份、CSV 文件名生成 |
| `dlkit.argparse_utils` | `GroupArgparse` | 多层级分组命令行参数解析器，支持 YAML/字典深度合并 |
| `dlkit.meters` | `ScalarMeter`, `TimeMeter` | 标量累加器与计时器 |
| `dlkit.tracker` | `CSVExpTracker` | CSV 实验记录管理：多服务器聚合、查重、统计 |
| `dlkit.summary` | `Summary` | JSON 格式标量指标记录 |
| `dlkit.utils` | `set_seed`, `count_parameters_in_MB`, `get_gpus_memory_info`, `cur_time_str`, `get_caller_filename` | 通用工具函数 |

## 快速开始

### 1. 使用 StandardTrainingPipeline

```python
from dlkit import StandardTrainingPipeline

config = {
    'train': {
        'lr': 0.01,
        'batch_size': 32,
    }
}

# 保留参数可直接作为关键字参数传入，获得 IDE 自动补全
pipeline = StandardTrainingPipeline(
    base_config=config,
    gpu=0,
    seed=42,
    platform='6001',
    workspace='experiments',   # 工作区根目录
    project_name='my_project', # 项目目录名
    csv_dir='./csv',           # CSV 文件保存目录
    csv_tag='search',          # CSV 文件名标签
)
args, config = pipeline.init(run_name='my_experiment')

pipeline.print('Training started', lr=args.train.lr)
pipeline.print(Epoch=1, Loss=0.45, Acc='92%')

# 自动生成 CSV 文件路径: './csv/train_search_6001.csv'
csv_path = pipeline.gen_csv_path(tag='search')

# 实验结束后可重命名目录
# pipeline.rename_run_name('Acc_92.5')
```

### 2. 使用 GroupArgparse

```python
from dlkit import GroupArgparse

config = {
    'model': {'type': 'resnet', 'layers': 50},
    'train': {'lr': 0.01, 'epochs': 100},
}

gparser = GroupArgparse(base_config=config)
gparser.set_cur_group('train')
gparser.add_argument('--lr', type=float, default=0.01)
gparser.add_argument('--epochs', type=int, default=100)

args = gparser.parse_args(export_dataclass=True)
print(args.train.lr)      # 支持点操作符访问
```

### 3. 使用 CSVExpTracker

```python
from dlkit import StandardTrainingPipeline, CSVExpTracker

# 方式一：从 args 自动推导 CSV 路径（推荐）
pipeline = StandardTrainingPipeline(
    platform='6001',
    csv_dir='./csv',
    csv_tag='search',
)
args, config = pipeline.init(run_name='my_experiment')

# 不传 root，自动从 args 读取 csv_dir / csv_tag / platform
# 生成路径: './csv/train_search_6001.csv'
tracker = CSVExpTracker(
    args=args,
    x_columns=['lr', 'batch_size'],
    y_columns=['acc', 'loss'],
    avoid_duplication=True,
    platforms=['6001', '6002'],
)

# 方式二：直接指定 root
# tracker = CSVExpTracker(root='./csv/results.csv', ...)

# 查重
if not tracker.is_sampled(lr=0.01, batch_size=32, run_id=0):
    # ... 训练模型 ...
    tracker.save_csv({'lr': 0.01, 'batch_size': 32, 'acc': 0.925, 'loss': 0.05, 'run_id': 0})

# 统计聚合
tracker.count()
```

### 4. 使用 Meters

```python
from dlkit import ScalarMeter, TimeMeter

loss_meter = ScalarMeter(reduction='mean')
timer = TimeMeter()

for batch in dataloader:
    timer.update()
    loss = train_batch(batch)
    loss_meter.update(loss, num=batch_size)

print(f'Avg Loss: {loss_meter.avg}, Time: {timer.cost} min')
```

### 5. 使用 get_caller_filename

```python
from dlkit import get_caller_filename

# 在 train.py 中调用
print(get_caller_filename())            # 'train'
print(get_caller_filename(stem=False))  # 'train.py'
```

## 相对原版 assistant.py 的修复

1. **`TimeMeter`** — 修复了 `start_time` 未初始化导致 `total` 属性崩溃的问题
2. **`set_seed`** — 使用 `manual_seed_all` 覆盖所有 GPU（原版仅 `manual_seed`）
3. **`CSVExpTracker.save_csv`** — 修复了文档声称更新 `self.df` 但实际未实现的问题
4. **`Summary.save_record`** — 修复了 `record_name=None` 时的逻辑不清晰问题
5. **`StandardTrainingPipeline`** — 修复了 `enable_directory=False` 时 `self.print` 未定义的问题
6. **清理** — 移除了废弃的 `_export_dataclass_schema1`、未使用的 `glob` import
7. **移除 checkpoint** — 移除了 `save_checkpoint` / `load_checkpoint`（原版存在死代码 bug，且该功能可由 PyTorch 原生 `torch.save` / `torch.load` 替代）
8. **新增 `get_caller_filename`** — 自动获取主调脚本文件名
9. **新增 `gen_csv_path`** — 自动生成 `[主调文件名]_[tag]_[platform].csv` 格式的 CSV 文件路径
10. **新增 `platform` 保留参数** — 用于标识当前运行所在的平台
11. **重命名目录参数** — `exp_root` → `workspace`（工作区根目录），`task_name` → `project_name`（项目名），含义更直观
12. **`CSVExpTracker` 支持 `args` 推导路径** — `root` 改为可选，未指定时自动从 `args` 中读取 `csv_dir`、`csv_tag`、`platform` 推导路径，默认保存到 `./csv/` 目录
13. **新增 `csv_dir`、`csv_tag` 保留参数** — 用于 `CSVExpTracker` 自动推导 CSV 路径

## 项目结构

```
dlkit/
├── pyproject.toml
├── README.md
└── src/
    └── dlkit/
        ├── __init__.py
        ├── py.typed
        ├── utils.py           # 通用工具函数 (含 get_caller_filename)
        ├── meters.py          # ScalarMeter, TimeMeter
        ├── tracker.py         # CSVExpTracker
        ├── summary.py         # Summary
        ├── argparse_utils.py  # GroupArgparse
        └── pipeline.py        # StandardTrainingPipeline (含 gen_csv_path)
```

## 发布到 PyPI

```bash
cd dlkit
pip install build twine
python -m build
twine upload dist/*
```

## License

MIT
