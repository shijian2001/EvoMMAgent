"""Crop tool for image region extraction."""

import json
from typing import Union, Dict

from tool.base_tool import BasicTool, register_tool
from tool.utils.image_utils import image_processing


@register_tool(name="crop")
class CropTool(BasicTool):
    """Crop an image with a bounding box."""
    
    name = "crop"
    description_en = "Crop a region from an image. Use localize_objects first if bbox is unknown."
    description_zh = "裁剪图像区域。若坐标未知，先用 localize_objects 定位。"
    
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
                "description": "Bounding box as [left, top, right, bottom], values between 0 and 1"
            }
        },
        "required": ["image", "bbox_2d"]
    }
    example = '{"image": "img_0", "bbox_2d": [0.1, 0.2, 0.5, 0.6]}'
    
    def call(self, params: Union[str, Dict]) -> Dict:
        params_dict = self.parse_params(params)
        
        image_path = params_dict["image"]
        bbox_2d = params_dict["bbox_2d"]
        
        # Validate bbox_2d
        if not all(0 <= x <= 1.0 for x in bbox_2d):
            return {"error": "bbox_2d values must be between 0 and 1"}
        if bbox_2d[0] >= bbox_2d[2] or bbox_2d[1] >= bbox_2d[3]:
            return {"error": "invalid bbox_2d: left >= right or top >= bottom"}
        
        # Load image
        try:
            image = image_processing(image_path)
        except Exception as e:
            return {"error": f"failed to load image: {e}"}
        
        # Crop
        W, H = image.size
        crop_box = (int(bbox_2d[0] * W), int(bbox_2d[1] * H), int(bbox_2d[2] * W), int(bbox_2d[3] * H))
        cropped = image.crop(crop_box)
        
        # Return dict with PIL Image object
        return {"output_image": cropped}
    
    def generate_description(self, properties, observation):
        """Generate description for cropped image."""
        img = properties.get("image", "image")
        bbox_2d = properties.get("bbox_2d", [])
        return f"Cropped {img} at bbox_2d {bbox_2d}"
