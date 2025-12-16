# Tool Integration Guide

## 快速开始

### 方式 1: Agent 使用（自动预加载）

```python
from agent.mm_agent import MultimodalAgent

# Agent 自动预加载 tool_bank 中的模型（默认检测所有 GPU）
agent = MultimodalAgent(
    tool_bank=["ocr", "localize_objects"],
    model_name="qwen2.5-vl-72b-instruct"
)

# 指定 GPU 分配
agent = MultimodalAgent(
    tool_bank=["ocr", "localize_objects"],
    model_name="qwen2.5-vl-72b-instruct",
    preload_devices=["cuda:0", "cuda:1"]  # 可选，默认自动检测
)

# 使用工具（无延迟，模型已预加载）
result = await agent.act(query="识别图片中的文字", images=["test.jpg"])
```

### 方式 2: 独立测试工具

```python
from tool import TOOL_REGISTRY
from pathlib import Path

# 创建输出目录
OUTPUT_DIR = Path("test_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# 调用工具（首次调用自动加载模型）
tool = TOOL_REGISTRY["localize_objects"]()
result = tool.call({"image": "test.jpg", "objects": ["dog"]})

# 手动保存多模态输出（PIL Image）
if "output_image" in result:
    from PIL import Image
    if isinstance(result["output_image"], Image.Image):
        result["output_image"].save(OUTPUT_DIR / "output.png")
```

### 方式 3: 手动预加载

```python
from tool.model_cache import preload_tools

# 预加载指定工具到指定 GPU
preload_tools(
    tool_bank=["ocr", "localize_objects"],
    devices=["cuda:0", "cuda:1"]  # 轮流分配
)

# 之后创建的工具实例会复用预加载的模型
```

## 1. 非模型工具 (Non-Model Tool)

适用于：计算器、图像处理、API 调用等不需要加载模型的工具。

```python
# tool/my_tool.py
import json
from typing import Union, Dict
from tool.base_tool import BasicTool, register_tool


@register_tool(name="my_tool")
class MyTool(BasicTool):
    name = "my_tool"
    description_en = "English description"
    description_zh = "中文描述"
    parameters = {
        "type": "object",
        "properties": {
            "param1": {"type": "string", "description": "参数说明"},
        },
        "required": ["param1"]
    }
    example = '{"param1": "value"}'

    def call(self, params: Union[str, Dict]) -> str:
        p = self.parse_params(params)
        
        # 实现逻辑
        result = process(p["param1"])
        
        return json.dumps({"success": True, "result": result})
```

## 2. 模型工具 (Model-Based Tool)

适用于：OCR、目标检测、分割等需要加载神经网络模型的工具。

```python
# tool/ocr_tool.py
import json
from typing import Union, Dict
from tool.base_tool import ModelBasedTool, register_tool


@register_tool(name="ocr")
class OCRTool(ModelBasedTool):
    name = "ocr"
    model_id = "ocr"
    
    description_en = "Extract text from image"
    description_zh = "从图像中提取文字"
    parameters = {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "图像路径"},
        },
        "required": ["image"]
    }
    example = '{"image": "image-0"}'

    def load_model(self, device: str):
        """加载模型并设置到 self.model"""
        import easyocr
        self.model = easyocr.Reader(["en"], gpu=device.startswith("cuda"))
        self.device = device
        self.is_loaded = True

    def _call_impl(self, params: Union[str, Dict]) -> str:
        """实现工具逻辑"""
        p = self.parse_params(params)
        result = self.model.readtext(p["image"])
        return json.dumps({"success": True, "text": result})
```

**约定**：主模型统一存储在 `self.model`，辅助组件（如 preprocess, tokenizer）可以自由命名。

## 3. 工具返回格式规范

### 统一返回格式：Dict

**所有工具必须返回 `Dict` 类型**，包含以下标准字段：

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `error` | `str` | 失败时必需 | 错误信息 |
| 数据字段 | Any | 成功时 | 工具返回的业务数据（语义化命名） |
| `output_image` | `PIL.Image` 或 `str` | 可选 | 图像输出（触发多模态处理） |
| `output_video` | `str` | 可选 | 视频输出（触发多模态处理） |

**关键原则：**
- ✅ **成功时**：只返回数据字段，无 `success` 字段
- ❌ **失败时**：只返回 `{"error": "..."}`
- 🎯 **数据字段命名**：语义化且简洁（如 `result`, `depth`, `objects`, `text`）
- 🚫 **避免冗余**：不返回输入参数的echo（如 `expression`, `mode`, `query`）

### 基本返回示例

#### 成功返回

```python
# 数据工具
@register_tool(name="calculator")
class CalculatorTool(BasicTool):
    def call(self, params):
        result = eval(expression)
        return {
            "success": True,
            "result": result,  # 业务数据
            "expression": expression
        }

# 图像工具
@register_tool(name="crop")
class CropTool(BasicTool):
    def call(self, params):
        cropped = image.crop(bbox)
        return {
            "success": True,
            "output_image": cropped,  # PIL.Image 对象（触发多模态处理）
            "original_size": [W, H],
            "cropped_size": [400, 300]
        }

# 视频工具
@register_tool(name="video_process")
class VideoProcessTool(BasicTool):
    def call(self, params):
        output_path = process_video(...)
        return {
            "success": True,
            "output_video": output_path,  # 文件路径（触发多模态处理）
            "duration": 10.5
        }
```

#### 失败返回

```python
# 所有工具的错误返回格式统一
def call(self, params):
    if validation_failed:
        return {
            "success": False,
            "error": "Invalid parameters: ..."
        }
    
    try:
        # ... 处理逻辑 ...
    except Exception as e:
        return {
            "success": False,
            "error": f"Error processing: {str(e)}"
        }
```

### Agent 自动处理

#### **纯数据工具**

Agent 将数据展开成自然语言句子传给 LLM：

```
# calculator 工具返回：{"result": 56088}
Observation: Result: 56088

# get_objects 工具返回：{"objects": ["cat", "dog", "tree"]}
Observation: Detected objects: cat, dog, tree

# ocr 工具返回：{"text": "Hello World"}
Observation: Extracted text: Hello World

# estimate_region_depth 工具返回：{"depth": 0.5432}
Observation: Estimated depth: 0.5432

# 错误情况返回：{"error": "bbox values must be between 0 and 1"}
Observation: bbox values must be between 0 and 1
```

#### **纯图像工具**

Agent 使用 Memory 生成的描述：

```
# crop 工具返回：{"output_image": <PIL.Image>, "original_size": [800, 600], "cropped_size": [400, 300]}
Observation: saved as img_1: Cropped img_0 at bbox [0.25, 0.25, 0.75, 0.75]

# zoom_in 工具返回：{"output_image": <PIL.Image>, "original_size": [800, 600], "zoomed_size": [1200, 900]}
Observation: saved as img_2: Zoomed in img_0 at bbox [0.5, 0.5, 0.7, 0.7] with factor 2.0
```

#### **图像+数据工具**

Agent 结合 Memory 描述和业务数据：

```
# localize_objects 工具返回：{"output_image": <PIL.Image>, "regions": [{"bbox": [...], "label": "dog"}, ...]}
Observation: saved as img_3: Localized regions on img_0. {"regions": [{"bbox": [0.1, 0.2, 0.3, 0.4], "label": "dog"}, {"bbox": [0.5, 0.6, 0.7, 0.8], "label": "cat"}]}

# detect_faces 工具返回：{"output_image": <PIL.Image>, "regions": [{"bbox": [...], "label": "face"}, ...]}
Observation: saved as img_4: Detected faces on img_0. {"regions": [{"bbox": [0.2, 0.1, 0.4, 0.5], "label": "face"}]}
```

#### **相似度工具**

Agent 展开成结构化句子：

```
# get_image2images_similarity 工具返回：{"similarity": [0.85, 0.72, 0.91], "best_image_index": 2}
Observation: Similarity scores: [0.85, 0.72, 0.91], best match at index 2
```

#### **处理流程**

1. **Memory 保存**：检测到 `output_image`/`output_video` 时，Memory 自动保存到 `memory/tasks/{task_id}/` 并生成 ID（如 `img_0`）
2. **描述生成**：Memory 根据工具类型和参数生成描述（如 "Cropped img_0 at bbox [...]"）
3. **数据组合**：Agent 将描述与其他业务数据（如 `regions`）组合传给 LLM

## 4. 测试工具

在 `test_tools.py` 添加测试：

```python
async def test():
    # 测试图像工具
    r = await TOOL_REGISTRY["localize_objects"]().call_async({
        "image": "test.jpg", 
        "objects": ["dog"]
    })
    # 保存多模态输出（自动检测 output_image 或 output_video）
    r = save_multimodal_output(r, "localize_objects")
    print(f"localize_objects: {r}")
    
    # 测试视频工具（未来）
    # r = await TOOL_REGISTRY["video_process"]().call_async({...})
    # r = save_multimodal_output(r, "video_process")  # 自动保存视频
```

运行：`python test_tools.py`

**输出说明**：
- 非模型工具：返回结果（JSON string 或 dict）
- 图像工具：返回 dict，包含 `output_image`（PIL.Image 对象或路径）
- 视频工具：返回 dict，包含 `output_video`（文件路径）
- `save_multimodal_output()` 自动保存到 `test_outputs/` 目录

---

## 注意事项

### 必须遵守

1. **返回格式**：返回 JSON 字符串，成功 `{"success": True, ...}`，失败 `{"error": "错误信息"}`
2. **参数验证**：使用 `self.parse_params(params)` 解析和验证参数
3. **必须有 example**：`example` 属性用于 prompt 中展示工具用法，Agent 依赖它来学习如何调用工具

### 模型工具注意事项

1. **主模型统一命名 `self.model`**：缓存和共享基于此约定

2. **辅助组件自由命名**：如 `self.preprocess`, `self.tokenizer` 等

3. **实现 `_call_impl` 而非 `call`**：`call` 自动处理模型加载和缓存

4. **GPU 自动选择**：不指定 device 时，系统自动选择可用 GPU

### 命名规范

| 类型 | 规范 | 示例 |
|------|------|------|
| 文件名 | `snake_case_tool.py` | `crop_tool.py`, `ocr_tool.py` |
| 类名 | `PascalCase` + `Tool` 后缀 | `CropTool`, `OCRTool` |
| 工具名 (name) | `snake_case` | `crop`, `ocr`, `solve_math` |
| 变量/函数 | `snake_case` | `output_path`, `load_model` |
| 常量 | `UPPER_SNAKE_CASE` | `OCR_MODEL_PATH` |

### 代码风格

- 保持简洁，避免冗余代码，英文注释
- 错误直接返回 `{"error": "..."}`，不要抛异常
- 使用类型注解 `Union[str, Dict]`
