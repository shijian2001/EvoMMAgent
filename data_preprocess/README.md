# 数据预处理

将下载到本地的数据集转换为统一的 parquet 格式，便于高效评测。

## 目录结构

```
data_preprocess/
├── explore_hf_dataset.py    # 通用探索工具（支持所有数据集类型）
├── image_dataset/           # 图像数据集预处理脚本
│   └── blink.py
└── video_dataset/           # 视频数据集预处理脚本（TODO）
```

## 使用流程

### 1. 探索数据集

先用探索工具查看数据集结构：

```bash
# 查看有哪些 split
python data_preprocess/explore_hf_dataset.py ~/data/blink --splits

# 查看 train split（显示 3 条样本）
python data_preprocess/explore_hf_dataset.py ~/data/blink train 3
```

**注意**：`explore_hf_dataset.py` 对图像和视频数据集都通用。

### 2. 编写预处理脚本

根据探索结果，在对应目录下编写预处理脚本：
- 图像数据集 → `image_dataset/`
- 视频数据集 → `video_dataset/`

### 3. 处理数据集

运行预处理脚本生成 parquet 文件。

## 统一格式

### 图像数据集格式

所有处理后的图像数据集遵循以下格式（所有字段都必须存在）：

| 字段 | 类型 | 说明 | 可否为空 |
|------|------|------|----------|
| `idx` | int | 样本索引 | ❌ |
| `images` | List[PIL.Image] | 图片列表 | ❌ 至少1张 |
| `dataset` | str | 数据集名称 | ❌ |
| `type` | str | 任务类型 | ❌ |
| `sub_task` | str | 子任务标识 | ✅ 可为空字符串 |
| `question` | str | 原始问题 | ❌ |
| `choices` | List[str] | 选项列表 | ✅ 可为空列表 |
| `answer` | str | 正确答案 | ❌ |
| `prompt` | str | 完整提问（含问题+选项） | ✅ 可为空字符串 |

**字段说明**：
- `question`: 原始问题文本
- `choices`: 选项列表（如 ["A. 选项1", "B. 选项2", ...]）
- `prompt`: 预先格式化的完整提问（包含 question 和 choices），可为空
  - 若为空，评测时可根据 question 和 choices 动态生成
  - 若非空，可直接使用预设的格式化提问

**任务类型 (`type`)**：
- `multi-choice`: 多选题
- `open-ended`: 开放式问答
- 其他自定义类型...

**关键点**：
- 所有字段都必须存在，但部分可以为空值
- `images` 使用 HuggingFace 的 `Image` Feature，自动处理编解码
- 加载后 `images` 已经是 `List[PIL.Image]`，无需手动解码

### 视频数据集格式

**TODO**: 视频数据集格式设计待定

## 使用处理后的数据

```python
import datasets

# 加载图像数据集
ds = datasets.load_dataset("parquet", data_files="~/data/blink/train.parquet", split="train")

# 单条样本
sample = ds[0]
print(sample["images"])    # 已经是 List[PIL.Image]
print(sample["question"])  # 原始问题
print(sample["choices"])   # 选项列表
print(sample["prompt"])    # 完整提问（可能为空）
print(sample["type"])      # 任务类型，如 "multi-choice"

# 批量迭代
for batch in ds.iter(batch_size=32):
    images = batch["images"]      # List[List[PIL.Image]]
    questions = batch["question"] # List[str]
    choices = batch["choices"]    # List[List[str]]
    prompts = batch["prompt"]     # List[str] - 可能为空
    answers = batch["answer"]     # List[str]
    types = batch["type"]         # List[str]
    
    # 评测逻辑
    # 使用预设 prompt 或动态生成
    final_prompts = [
        p if p else format_prompt(q, c)
        for p, q, c in zip(prompts, questions, choices)
    ]
    predictions = model(images, final_prompts)
    evaluate(predictions, answers)

# 按任务类型过滤
multi_choice = ds.filter(lambda x: x["type"] == "multi-choice")
```