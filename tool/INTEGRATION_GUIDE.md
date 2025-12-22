# Tool Integration Guide

## 快速开始

### 方式 1: Agent 使用（推荐）

```python
from agent.mm_agent import MultimodalAgent

# Agent 自动预加载模型（默认检测所有 GPU）
agent = MultimodalAgent(
    tool_bank=["ocr", "localize_objects"],
    model_name="qwen2.5-vl-72b-instruct"
)

# 使用工具（无延迟，模型已预加载）
result = await agent.act(query="识别图片中的文字", images=["test.jpg"])
```

### 方式 2: 独立测试

```python
from tool import TOOL_REGISTRY

# 调用工具（首次调用自动加载模型）
tool = TOOL_REGISTRY["localize_objects"]()
result = tool.call({"image": "test.jpg", "objects": ["dog"]})
```

---

## ⚠️ Image 参数设计

**对外**：用户传 ID（`"img_0"`）  
**内部**：工具收路径（`"/path/to/image.png"`）  
**转换**：`memory.resolve_ids()` 自动处理

```python
# 参数定义：对外写 ID
parameters = {
    "properties": {
        "image": {"type": "string", "description": "Image ID (e.g., 'img_0')"}
    }
}

# 工具实现：内部是路径
def call(self, params):
    image_path = params["image"]  # 已是路径，直接用
    image = image_processing(image_path)
```

**不要在工具内部手动解析 ID**，Memory 已自动处理。

---

## 工具类型

### 1. 非模型工具

无需加载模型的工具（计算器、图像裁剪等）。

```python
from tool.base_tool import BasicTool, register_tool

@register_tool(name="calculator")
class CalculatorTool(BasicTool):
    name = "calculator"
    description_en = "Perform arithmetic calculations"
    description_zh = "执行算术计算"
    parameters = {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Math expression"}
        },
        "required": ["expression"]
    }
    example = '{"expression": "123 * 456"}'
    
    def call(self, params):
        p = self.parse_params(params)
        result = eval(p["expression"])
        return {"result": result}
```

### 2. 模型工具

需要加载神经网络模型的工具（OCR、检测等）。

```python
from tool.base_tool import ModelBasedTool, register_tool

@register_tool(name="ocr")
class OCRTool(ModelBasedTool):
    name = "ocr"
    model_id = "ocr"  # 缓存标识
    
    description_en = "Extract text from image"
    description_zh = "从图像中提取文字"
    parameters = {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "Image ID (e.g., 'img_0')"}
        },
        "required": ["image"]
    }
    example = '{"image": "img_0"}'
    
    def load_model(self, device: str):
        """加载模型（只执行一次）"""
        import easyocr
        self.model = easyocr.Reader(["en"], gpu=device.startswith("cuda"))
        self.device = device
        self.is_loaded = True
    
    def _call_impl(self, params):
        """实现工具逻辑（每次调用执行）"""
        p = self.parse_params(params)
        result = self.model.readtext(p["image"])
        return {"text": result}
```

**约定**：主模型存 `self.model`，其他组件（processor, tokenizer）随意命名。

### 🔥 自动缓存机制

**模型组件自动缓存，无需手动管理！**

```python
def load_model(self, device: str):
    self.model = ...          # ✅ 自动缓存
    self.processor = ...      # ✅ 自动缓存
    self.tokenizer = ...      # ✅ 自动缓存
    self.preprocess = ...     # ✅ 自动缓存
    self._temp_data = ...     # ❌ 不缓存（私有属性）
```

**规则**：
- 公开属性 `self.xxx` → 自动缓存
- 私有属性 `self._xxx` → 不缓存
- 状态属性（device, is_loaded）→ 自动排除

**好处**：多个 Agent 共享同一模型实例，10 个 Agent = 1 份模型内存。

---

## 返回格式

### 统一格式：Dict

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `error` | str | 失败时 | 错误信息 |
| 数据字段 | Any | 成功时 | 业务数据（语义化命名） |
| `output_image` | PIL.Image / str | 可选 | 图像输出 |
| `output_video` | str | 可选 | 视频输出 |

**原则**：
- ✅ 成功：只返回数据字段
- ❌ 失败：只返回 `{"error": "..."}`
- 🎯 数据字段：语义化命名（`result`, `objects`, `text`）
- 🚫 不返回输入参数的 echo

### 示例

```python
# 成功 - 数据工具
return {"result": 56088}

# 成功 - 图像工具
return {
    "output_image": cropped_img,  # PIL.Image 对象
    "original_size": [800, 600],
    "cropped_size": [400, 300]
}

# 成功 - 图像+数据工具
return {
    "output_image": annotated_img,
    "regions": [{"bbox": [0.1, 0.2, 0.3, 0.4], "label": "dog"}]
}

# 失败
return {"error": "Invalid bbox values"}
```

### Agent 如何处理

**纯数据** → 展开成句子：
```
{"result": 56088} → "Result: 56088"
{"objects": ["cat", "dog"]} → "Detected objects: cat, dog"
```

**纯图像** → 使用 Memory 描述：
```
{"output_image": <PIL.Image>} → "saved as img_1: Cropped img_0 at bbox [...]"
```

**图像+数据** → 组合描述和数据：
```
{"output_image": <PIL.Image>, "regions": [...]} 
→ "saved as img_3: Localized regions on img_0. {'regions': [...]}"
```

---

## 测试工具

```python
# run_tool_test.py
from tool import TOOL_REGISTRY

async def test():
    tool = TOOL_REGISTRY["ocr"]()
    result = await tool.call_async({"image": "test.jpg"})
    print(result)
```

运行：`python run_tool_test.py`

---

## 核心规范

### 必须遵守

1. **返回 Dict**：成功返回数据字段，失败返回 `{"error": "..."}`
2. **参数验证**：使用 `self.parse_params(params)` 解析
3. **必须有 example**：Agent 依赖它学习调用方式

### 模型工具

1. 主模型命名 `self.model`（缓存约定）
2. 实现 `_call_impl` 而非 `call`（自动处理加载）
3. GPU 自动选择（不指定 device）

### 命名规范

| 类型 | 规范 | 示例 |
|------|------|------|
| 文件名 | `snake_case_tool.py` | `crop_tool.py` |
| 类名 | `PascalCase` + `Tool` | `CropTool` |
| 工具名 | `snake_case` | `crop`, `ocr` |

---

## 工具列表

### 非模型工具

| 工具名 | 描述 | 输入 | 输出 |
|--------|------|------|------|
| `calculator` | 算术计算 | expression | result |
| `solve_math_equation` | 解方程 | equation, variable | solution |
| `crop` | 裁剪图像 | image, bbox | output_image |
| `zoom_in` | 放大图像 | image, bbox, zoom_factor | output_image |

### 模型工具

| 工具名 | 描述 | 模型 | 输入 | 输出 |
|--------|------|------|------|------|
| `ocr` | 文字识别 | EasyOCR | image | text |
| `get_objects` | 目标检测 | OWLv2 | image | objects |
| `localize_objects` | 目标定位 | OWLv2 | image, objects | output_image, regions |
| `detect_faces` | 人脸检测 | MTCNN | image | output_image, regions |
| `estimate_region_depth` | 区域深度 | Depth-Anything | image, bbox | depth |
| `estimate_object_depth` | 物体深度 | Depth-Anything + OWLv2 | image, object | depth |
| `get_image2texts_similarity` | 图文相似度 | CLIP | image, texts | similarity, best_text_index |
| `get_image2images_similarity` | 图图相似度 | CLIP | image, candidate_images | similarity, best_image_index |
| `get_text2images_similarity` | 文图相似度 | CLIP | text, images | similarity, best_image_index |
| `visualize_regions` | 可视化区域 | - | image, regions | output_image |

---

## 常见问题

### Q: 如何添加新工具？

1. 创建 `tool/my_tool.py`
2. 继承 `BasicTool` 或 `ModelBasedTool`
3. 使用 `@register_tool(name="my_tool")` 装饰器
4. 在 `tool/__init__.py` 中导入

### Q: 模型缓存如何工作？

- 首个 Agent 加载模型并缓存
- 后续 Agent 复用缓存（零延迟）
- 进程结束时自动释放

### Q: 如何处理图像路径？

使用 `image_processing()` 工具函数：
```python
from tool.utils.image_utils import image_processing
image = image_processing(image_path)  # 返回 PIL.Image
```

### Q: 工具调用失败怎么办？

返回错误信息：
```python
try:
    result = process(params)
    return {"result": result}
except Exception as e:
    return {"error": f"Processing failed: {str(e)}"}
```
