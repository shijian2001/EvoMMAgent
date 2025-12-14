"""Tool for localizing objects in images with bounding boxes using transformers Grounding DINO."""

import json
from typing import Union, Dict, List
import torch
from PIL import Image
from tool.base_tool import ModelBasedTool, register_tool
from tool.utils.image_utils import image_processing
from tool.visualize_regions_tool import VisualizeRegionsOnImageTool

@register_tool(name="localize_objects")
class LocalizeObjectsTool(ModelBasedTool):
    name = "localize_objects"
    description_en = "Localize one or multiple objects/regions with bounding boxes. This tool may output objects that don't exist or miss objects that do. You should use the output only as weak evidence for reference. When answering questions about the image, you should double-check the detected objects. You should be especially cautious about the total number of regions detected, which can be more or less than the actual number."
    description_zh = "定位图像中的一个或多个对象/区域，并返回边界框。此工具可能输出不存在的对象或遗漏存在的对象。输出仅作为弱证据参考。回答图像问题时，应仔细检查检测到的对象。特别要注意检测到的区域总数，可能多于或少于实际数量。"
    parameters = {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "The image to localize objects/regions in"},
            "objects": {"type": "array", "items": {"type": "string"}, "description": "A list of object names to localize. e.g. ['dog', 'cat', 'person']. The model might not be able to detect rare objects or objects with complex descriptions."}
        },
        "required": ["image", "objects"]
    }
    example = '{"image": "image-0", "objects": ["dog", "cat"]}'

    def load_model(self, device: str) -> None:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        from accelerate import Accelerator
        from tool.model_config import GROUNDING_DINO_WEIGHTS_PATH
        model_id = GROUNDING_DINO_WEIGHTS_PATH
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        self.device = device
        self.is_loaded = True

    def _call_impl(self, params: Union[str, Dict]) -> str:
        params_dict = self.parse_params(params)
        image_path = params_dict["image"]
        objects = params_dict["objects"]
        if not isinstance(objects, list) or len(objects) == 0:
            return json.dumps({"success": False, "error": "objects must be a non-empty list of strings"})
        try:
            # 加载图片
            image_path_full = image_processing(image_path, return_path=True)
            original_image = image_processing(image_path)
            image = Image.open(image_path_full).convert("RGB")
            text_labels = [objects]
            # 处理输入
            inputs = self.processor(images=image, text=text_labels, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=0.4,
                text_threshold=0.3,
                target_sizes=[image.size[::-1]]
            )
            result = results[0]
            regions = []
            obj_cnt = {}
            for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
                bbox = [round(float(x), 4) for x in box.tolist()]
                label_str = str(label)
                obj_cnt[label_str] = obj_cnt.get(label_str, 0) + 1
                label_out = f"{label_str}-{obj_cnt[label_str]}" if obj_cnt[label_str] > 1 else label_str
                regions.append({
                    "label": label_out,
                    "bbox": bbox,
                    "score": round(score.item(), 4)
                })
            # 可视化
            visualize_tool = VisualizeRegionsOnImageTool()
            visualize_params = {
                "image_path": image_path_full,
                "regions": [{"bbox": r["bbox"], "label": r["label"]} for r in regions]
            }
            output_image = visualize_tool.call(visualize_params)
            
            # Return dict with PIL Image
            return {
                "success": True,
                "output_image": output_image,  # PIL.Image object
                "regions": regions
            }
        except FileNotFoundError as e:
            return json.dumps({"success": False, "error": f"Image file not found: {str(e)}"})
        except Exception as e:
            return json.dumps({"success": False, "error": f"Error localizing objects: {str(e)}"})

