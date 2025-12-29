"""Zoom in tool for image region magnification."""

import json
from typing import Union, Dict
from PIL import Image

from tool.base_tool import BasicTool, register_tool
from tool.utils.image_utils import image_processing


@register_tool(name="zoom_in")
class ZoomInTool(BasicTool):
    """Zoom in on a region of an image."""
    
    name = "zoom_in"
    description_en = "Zoom in on a region of the image by cropping and resizing."
    description_zh = "裁剪并放大图像的指定区域。"
    
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
                "description": "Bounding box [left, top, right, bottom], values between 0 and 1"
            },
            "zoom_factor": {
                "type": "number",
                "description": "Zoom factor (must be > 1)",
                "exclusiveMinimum": 1.0
            }
        },
        "required": ["image", "bbox_2d", "zoom_factor"]
    }
    example = '{"image": "img_0", "bbox_2d": [0.25, 0.25, 0.75, 0.75], "zoom_factor": 2.0}'
    
    def call(self, params: Union[str, Dict]) -> Dict:
        params_dict = self.parse_params(params)
        
        image_path = params_dict["image"]
        bbox_2d = params_dict["bbox_2d"]
        zoom_factor = params_dict["zoom_factor"]
        
        # Validate
        if not all(0 <= x <= 1.0 for x in bbox_2d):
            return {"error": "bbox_2d values must be between 0 and 1"}
        if bbox_2d[0] >= bbox_2d[2] or bbox_2d[1] >= bbox_2d[3]:
            return {"error": "invalid bbox_2d: left >= right or top >= bottom"}
        if zoom_factor <= 1:
            return {"error": "zoom_factor must be > 1"}
        
        # Load image
        try:
            image = image_processing(image_path)
        except Exception as e:
            return {"error": f"failed to load image: {e}"}
        
        # Crop
        W, H = image.size
        crop_box = (int(bbox_2d[0] * W), int(bbox_2d[1] * H), int(bbox_2d[2] * W), int(bbox_2d[3] * H))
        cropped = image.crop(crop_box)
        
        # Zoom (resize)
        new_size = (int(cropped.width * zoom_factor), int(cropped.height * zoom_factor))
        zoomed = cropped.resize(new_size, Image.LANCZOS)
        
        # Return dict with PIL Image object
        return {"output_image": zoomed}
    
    def generate_description(self, properties, observation):
        """Generate description for zoomed image."""
        img = properties.get("image", "image")
        bbox_2d = properties.get("bbox_2d", [])
        factor = properties.get("factor", 2)
        return f"Zoomed in {img} at bbox_2d {bbox_2d} by {factor}x"
