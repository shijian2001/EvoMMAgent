"""Crop tool for image region extraction."""

import json
from typing import Union, Dict

from tool.base_tool import BasicTool, register_tool
from tool.utils.image_utils import image_processing


@register_tool(name="crop")
class CropTool(BasicTool):
    """Crop an image with a bounding box."""
    
    name = "crop"
    description_en = "Crop an image region specified by a bounding box."
    description_zh = "使用边界框裁剪图像区域。"
    
    parameters = {
        "type": "object",
        "properties": {
            "image": {
                "type": "string",
                "description": "Image ID (e.g., 'img_0')"
            },
            "bbox": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4,
                "description": "Bounding box as [left, top, right, bottom], values between 0 and 1"
            }
        },
        "required": ["image", "bbox"]
    }
    example = '{"image": "img_0", "bbox": [0.1, 0.2, 0.5, 0.6]}'
    
    def call(self, params: Union[str, Dict]) -> Dict:
        params_dict = self.parse_params(params)
        
        image_path = params_dict["image"]
        bbox = params_dict["bbox"]
        
        # Validate bbox
        if not all(0 <= x <= 1.0 for x in bbox):
            return {"error": "bbox values must be between 0 and 1"}
        if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            return {"error": "invalid bbox: left >= right or top >= bottom"}
        
        # Load image
        try:
            image = image_processing(image_path)
        except Exception as e:
            return {"error": f"failed to load image: {e}"}
        
        # Crop
        W, H = image.size
        crop_box = (int(bbox[0] * W), int(bbox[1] * H), int(bbox[2] * W), int(bbox[3] * H))
        cropped = image.crop(crop_box)
        
        # Return dict with PIL Image object
        return {"output_image": cropped}
