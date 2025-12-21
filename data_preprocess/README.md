# 数据预处理

将 HuggingFace 数据集转换为统一的 **JSONL + 图像文件夹** 格式。

## 目录结构

```
data_preprocess/
├── explore_hf_data.py    # 数据集探索工具
└── image/
    └── blink.py          # BLINK 数据集预处理
```

## 使用流程

### 1. 探索数据集

```bash
python data_preprocess/explore_hf_data.py ~/data/blink
```

### 2. 处理数据集

参考 `image/blink.py` 编写对应数据集的处理脚本：

```bash
python data_preprocess/image/blink.py ~/data/blink \
    --jsonl_path ~/output/blink/data.jsonl \
    --image_dir ~/output/blink/images
```

**输出结构**：
```
output/
├── data.jsonl          # 数据索引
└── images/             # 图像文件
    ├── 00000/
    │   ├── 00000.png
    │   └── 00001.png
    └── 00001/
        └── 00000.png
```

### 3. 编写新数据集处理脚本

参考 `image/blink.py` 的结构，需实现三个核心函数：

```python
def load_dataset(input_dir: str):
    """加载源数据集"""
    pass

def extract_images(example: dict) -> list:
    """从样本中提取 PIL 图像列表"""
    pass

def convert_sample(example: dict, idx: int, image_paths: list[str]) -> dict:
    """转换样本为统一格式（见下文）"""
    pass
```

通用函数 `save_images()` 和 `process_and_save()` 可直接复用。

## 统一格式

JSONL 中每行为一条记录，包含以下字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `idx` | int | 样本索引 |
| `images` | List[str] | 图像相对路径（如 `["00001/00000.png"]`） |
| `dataset` | str | 数据集名称 |
| `type` | str | 任务类型（`multi-choice`/`open-ended`） |
| `sub_task` | str | 子任务标识（可为空） |
| `question` | str | 问题文本 |
| `choices` | List[str] | 选项列表（可为空） |
| `answer` | str | 正确答案 |
| `prompt` | str | 完整提问（可为空） |

**示例**：
```json
{"idx": 0, "images": ["00000/00000.png", "00000/00001.png"], "dataset": "BLINK", "type": "multi-choice", ...}
```

## 加载数据

```python
import json
from pathlib import Path
from PIL import Image

def load_data(jsonl_path: str, image_dir: str):
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            record = json.loads(line)
            # 加载图像
            record['images'] = [
                Image.open(Path(image_dir) / img_path)
                for img_path in record['images']
            ]
            data.append(record)
    return data
```