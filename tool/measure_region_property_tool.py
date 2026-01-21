"""Tool for measuring visual properties of a region in an image."""

import numpy as np
from typing import Union, Dict, List

from tool.base_tool import BasicTool, register_tool
from tool.utils.image_utils import image_processing


@register_tool(name="measure_region_property")
class MeasureRegionPropertyTool(BasicTool):
    """Measure visual properties (brightness, contrast, saturation) of an image region."""
    
    name = "measure_region_property"
    description_en = "Measure brightness, contrast, and saturation of a region. Use localize_objects first if bbox is unknown."
    description_zh = "测量区域的亮度、对比度和饱和度。若坐标未知，先用 localize_objects 定位。"
    
    parameters = {
        "type": "object",
        "properties": {
            "image": {
                "type": "string",
                "description": "Image ID (e.g., 'img_0')"
            },
            "bbox_2d": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4,
                "description": "Bounding box [x1, y1, x2, y2] in normalized coordinates (0-1). If not provided, measures the entire image."
            },
            "properties": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["brightness", "contrast", "saturation"]
                },
                "description": "List of properties to measure. Default: all properties."
            }
        },
        "required": ["image"]
    }
    example = '{"image": "img_0", "bbox_2d": [0.1, 0.2, 0.5, 0.6], "properties": ["brightness", "saturation"]}'
    
    def call(self, params: Union[str, Dict]) -> Dict:
        params_dict = self.parse_params(params)
        
        image_path = params_dict["image"]
        bbox_2d = params_dict.get("bbox_2d")
        props = params_dict.get("properties", ["brightness", "contrast", "saturation"])
        
        # Validate bbox_2d if provided
        if bbox_2d:
            if not all(0 <= x <= 1.0 for x in bbox_2d):
                return {"error": "bbox_2d values must be between 0 and 1"}
            if bbox_2d[0] >= bbox_2d[2] or bbox_2d[1] >= bbox_2d[3]:
                return {"error": "invalid bbox_2d: left >= right or top >= bottom"}
        
        try:
            image = image_processing(image_path)
            W, H = image.size
            
            # Extract region
            if bbox_2d:
                x1, y1 = int(bbox_2d[0] * W), int(bbox_2d[1] * H)
                x2, y2 = int(bbox_2d[2] * W), int(bbox_2d[3] * H)
                region = image.crop((x1, y1, x2, y2))
            else:
                region = image
            
            # Convert to numpy array
            rgb = np.array(region, dtype=np.float32)
            
            results = {}
            
            if "brightness" in props:
                # Luminance using BT.601 coefficients
                luminance = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]
                results["brightness"] = round(float(np.mean(luminance)) / 255.0, 4)
            
            if "contrast" in props:
                # Standard deviation of luminance (normalized)
                luminance = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]
                results["contrast"] = round(float(np.std(luminance)) / 255.0, 4)
            
            if "saturation" in props:
                # HSV saturation: S = (max - min) / max
                r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
                max_rgb = np.maximum(np.maximum(r, g), b)
                min_rgb = np.minimum(np.minimum(r, g), b)
                # Avoid division by zero
                saturation = np.where(max_rgb > 0, (max_rgb - min_rgb) / max_rgb, 0)
                results["saturation"] = round(float(np.mean(saturation)), 4)
            
            return results
            
        except FileNotFoundError as e:
            return {"error": f"Image file not found: {str(e)}"}
        except Exception as e:
            return {"error": f"Error measuring region properties: {str(e)}"}
