# Tool Integration Guide

## 快速开始

### Agent 自动预加载（推荐）

```python
from agent.mm_agent import MultimodalAgent

# Agent 根据 tool_bank 自动预加载模型
agent = MultimodalAgent(
    tool_bank=["ocr", "localize_objects", "get_text2images_similarity"],
    preload_tools=True  # 默认开启
)
```

### 手动控制预加载

```python
from tool.model_config import preload_tools

# 预加载指定工具
preload_tools(tool_bank=["ocr", "localize_objects"])

# 指定 GPU 分配
preload_tools(tool_bank=["ocr", "clip"], devices=["cuda:0", "cuda:1"])
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

**✨ 自动模型共享 + 自动解包**：设置 `model_id`，系统自动加载和解包模型！

```python
# tool/ocr_tool.py
import json
from typing import Union, Dict
from tool.base_tool import ModelBasedTool, register_tool


@register_tool(name="ocr")
class OCRTool(ModelBasedTool):
    name = "ocr"
    model_id = "ocr"  # 引用 model_config.py 中的模型
    
    description_en = "Extract text from image"
    description_zh = "从图像中提取文字"
    parameters = {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "图像路径"},
        },
        "required": ["image"]
    }
    example = '{"image": "/path/to/image.jpg"}'

    def _call_impl(self, params: Union[str, Dict]) -> str:
        """实现工具逻辑（模型已自动加载到 self.reader）"""
        p = self.parse_params(params)
        result = self.reader.readtext(p["image"])  # self.reader 自动可用！
        return json.dumps({"success": True, "text": result})
```

**无需实现 `load_model_components`**！模型组件根据 `model_config.py` 中的配置自动解包到对应属性。

### 2.1 添加新模型到 model_config.py

在 `tool/model_config.py` 中添加模型定义：

```python
# 1. 添加模型路径常量
YOUR_MODEL_PATH = "/path/to/model"

# 2. 定义模型加载函数（可以返回单个模型或元组）
def _load_your_model(device: str):
    """Load your model."""
    from some_library import load_model
    model = load_model(YOUR_MODEL_PATH)
    model.to(device)
    return model  # 单个模型，或返回 (model, processor) 元组

# 3. 注册到 MODEL_REGISTRY（包含属性映射）
MODEL_REGISTRY = {
    "clip": {
        "loader": _load_clip_model,
        "attrs": ["model", "preprocess", "tokenizer"]  # 自动解包到这些属性
    },
    "your_model": {
        "loader": _load_your_model,
        "attrs": ["model"]  # 单个模型，或 ["model", "processor"] 多个组件
    },
}
```

**工具中直接使用对应属性**：设置 `model_id = "your_model"` 后，自动可用 `self.model`！

### 2.2 工具初始化（关键）

#### 方式 1：Agent 自动预加载（推荐）

```python
from agent.mm_agent import MultimodalAgent

# Agent 根据 tool_bank 自动预加载需要的模型
agent = MultimodalAgent(
    tool_bank=["ocr", "localize_objects", "get_text2images_similarity"],
    model_name="qwen2.5-vl-72b-instruct",
    preload_tools=True,  # 默认开启
    preload_devices=["cuda:0", "cuda:1"]  # 可选，默认自动检测
)

# 工具已预加载，首次调用无延迟
result = await agent.act(query="What's in the image?", images=["test.jpg"])
```

#### 方式 2：手动预加载

```python
from tool.model_config import preload_tools

# 预加载指定工具
preload_tools(tool_bank=["ocr", "localize_objects", "clip"])

# 预加载所有工具
preload_tools()

# 指定 GPU 分配
preload_tools(tool_bank=["ocr", "clip"], devices=["cuda:0", "cuda:1"])
```

#### 工作原理

- **按需加载**：只加载 `tool_bank` 中工具需要的模型
- **智能分配**：一个模型一个 GPU（轮流分配）
- **自动共享**：多个工具使用同一 `model_id` 时只加载一次
- **即时可用**：预加载后工具调用无延迟

#### 输出示例

```
🚀 Preloading 3 models for 5 tools across 2 device(s)...
  ✓ clip                -> cuda:0
  ✓ grounding_dino      -> cuda:1
  ✓ ocr                 -> cuda:0
```

### 2.3 多模型工具（高级）

单个工具使用多个模型时，将 `model_id` 设为列表：

```python
@register_tool(name="hybrid_tool")
class HybridTool(ModelBasedTool):
    model_id = ["clip", "grounding_dino"]  # 多个模型
    
    def _call_impl(self, params):
        # 所有模型组件自动可用！
        # CLIP: self.model, self.preprocess, self.tokenizer
        # GroundingDINO: self.model, self.processor (同名覆盖，后者生效)
        ...
```

**注意**：多个模型有同名属性时会覆盖。如需区分，在 `model_config.py` 中设置不同的属性名。

## 3. 文件输出（图像/视频工具）

参考 `crop_tool.py`, 对于返回图像、视频等文件的工具，使用 `TempManager` 保存到 `memory/` 目录：

```python
from tool.utils.temp_manager import get_temp_manager

# 获取输出路径：memory/工具名/输入文件名_后缀.扩展名
output_path = get_temp_manager().get_output_path(
    "tool_name",      # 工具名，会创建子目录
    input_path,       # 输入文件路径，用于生成输出文件名
    "suffix"          # 后缀，如 "cropped", "zoomed"
)
image.save(output_path)

# 返回时包含输出路径
return json.dumps({
    "success": True,
    "output_image": output_path,
    ...
})
```

## 4. 测试工具

在 `test_tools.py` 添加测试：

```python
r = await TOOL_REGISTRY["my_tool"]().call_async({"param1": "test"})
print(f"my_tool: {r}")
```

运行：`python test_tools.py`

> 对于图像/视频工具，运行后检查 `memory/工具名/` 目录下的输出文件是否正确。

---

## 注意事项

### 必须遵守

1. **返回格式**：返回 JSON 字符串，成功 `{"success": True, ...}`，失败 `{"error": "错误信息"}`
2. **参数验证**：使用 `self.parse_params(params)` 解析和验证参数
3. **必须有 example**：`example` 属性用于 prompt 中展示工具用法，Agent 依赖它来学习如何调用工具

### 模型工具特别注意

1. **只需设置 `model_id`**：模型加载、解包、GPU 分配全自动

2. **模型组件自动可用**：根据 `model_config.py` 中的 `attrs` 配置自动解包
   - `"ocr"` → `self.reader`
   - `"clip"` → `self.model`, `self.preprocess`, `self.tokenizer`
   - `"grounding_dino"` → `self.model`, `self.processor`

3. **实现 `_call_impl` 而非 `call`**：`call` 自动处理模型加载

4. **无需实现 `load_model_components`**：系统自动解包

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
