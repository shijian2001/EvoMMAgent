"""Tool for localizing objects in images with bounding boxes using SAM3."""

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
    model_id = "sam3"
    
    description_en = "Localize objects and return their bounding boxes. Use this to get bbox for other region-based tools."
    description_zh = "定位对象并返回边界框。可为其他需要坐标的工具提供输入。"
    parameters = {
        "type": "object",
        "properties": {
            "image": {"type": "string", "description": "Image ID (e.g., 'img_0')"},
            "objects": {"type": "array", "items": {"type": "string"}, "description": "A list of object names to localize. e.g. ['dog', 'cat', 'person']."}
        },
        "required": ["image", "objects"]
    }
    example = '{"image": "img_0", "objects": ["dog", "cat"]}'

    def load_model(self, device: str) -> None:
        from transformers import Sam3Processor, Sam3Model
        from tool.model_config import SAM3_MODEL_PATH
        self.processor = Sam3Processor.from_pretrained(SAM3_MODEL_PATH)
        self.model = Sam3Model.from_pretrained(SAM3_MODEL_PATH).to(device)
        self.device = device
        self.is_loaded = True

    def _call_impl(self, params: Union[str, Dict]) -> str:
        params_dict = self.parse_params(params)
        image_path = params_dict["image"]
        objects = params_dict["objects"]
        if not isinstance(objects, list) or len(objects) == 0:
            return {
                "error": "objects must be a non-empty list of strings"
            }
        try:
            image = image_processing(image_path)
            W, H = image.size
            
            regions = []
            obj_cnt = {}
            
            # Process each object separately with SAM3
            for obj_name in objects:
                # Prepare inputs for SAM3
                inputs = self.processor(images=image, text=obj_name, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Post-process to get instance segmentation results
                results = self.processor.post_process_instance_segmentation(
                    outputs,
                    threshold=0.5,
                    mask_threshold=0.5,
                    target_sizes=inputs.get("original_sizes").tolist()
                )[0]
                
                boxes = results["boxes"]
                scores = results["scores"]
                
                # Filter by score > 0.50 and process results
                for box, score in zip(boxes, scores):
                    score_val = score.item()
                    if score_val < 0.50:
                        continue
                    
                    # Convert box to list and normalize to [0, 1] range
                    box_list = box.tolist()
                    bbox = [
                        round(float(box_list[0]) / W, 4),  # x1
                        round(float(box_list[1]) / H, 4),  # y1
                        round(float(box_list[2]) / W, 4),  # x2
                        round(float(box_list[3]) / H, 4)   # y2
                    ]
                    
                    obj_cnt[obj_name] = obj_cnt.get(obj_name, 0) + 1
                    label_out = f"{obj_name}-{obj_cnt[obj_name]}" if obj_cnt[obj_name] > 1 else obj_name
                    regions.append({
                        "label": label_out,
                        "bbox_2d": bbox,
                        "score": round(score_val, 4)
                    })
            
            # Visualize results
            visualize_tool = VisualizeRegionsOnImageTool()
            visualize_params = {
                "image": image_path,
                "regions": [{"bbox_2d": r["bbox_2d"], "label": r["label"]} for r in regions]
            }
            output_image_result = visualize_tool.call(visualize_params)
            output_image = output_image_result.get("output_image")
            
            # Return dict with PIL Image
            return {
                "output_image": output_image,
                "regions": regions
            }
        except FileNotFoundError as e:
            return {"error": f"Image file not found: {str(e)}"}
        except Exception as e:
            return {"error": f"Error localizing objects: {str(e)}"}
    
    def generate_description(self, properties, observation):
        """Generate description for localized objects."""
        img = properties.get("image", "image")
        objects = properties.get("objects", [])
        if isinstance(objects, list):
            objects_str = ", ".join(objects) if objects else "objects"
        else:
            objects_str = str(objects)
        return f"Localized {objects_str} in {img}"