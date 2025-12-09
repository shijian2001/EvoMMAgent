# Tool Integration Guide

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

    def load_model(self, device: str) -> None:
        """加载模型到指定设备，首次调用时自动触发"""
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained("model_name").to(device)
        self.device = device
        self.is_loaded = True

    def _call_impl(self, params: Union[str, Dict]) -> str:
        """模型加载后执行的实际逻辑"""
        p = self.parse_params(params)
        result = self.model.predict(p["image"])
        return json.dumps({"success": True, "text": result})
```

## 3. 文件输出（图像/视频工具）

对于返回图像、视频等文件的工具，使用 `TempManager` 保存到 `memory/` 目录：

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

1. **load_model 必须设置三个属性**：
   - `self.model` - 模型实例
   - `self.device` - 当前设备
   - `self.is_loaded = True` - 标记已加载

2. **GPU 自动管理**：不需要手动选择 GPU，系统会自动选择空闲显卡

3. **实现 `_call_impl` 而非 `call`**：`call` 会自动处理模型加载

### 代码风格

- 保持简洁，避免冗余代码，英文注释
- 错误直接返回 `{"error": "..."}`，不要抛异常
- 使用类型注解 `Union[str, Dict]`
