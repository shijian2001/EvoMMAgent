"""Crop tool for image region extraction."""

import ast
import json
from typing import Union, Dict

from tool.base_tool import BasicTool, register_tool
from tool.utils.image_utils import image_processing, expand_bbox, visualize_bbox


@register_tool(name="crop")
class CropTool(BasicTool):
    """Crop an image with a bounding box.
    
    Labels the cropped region with a bounding box and crops the region 
    with margins around the bbox for contextual understanding.
    """
    
    name = "crop"
    description_en = (
        "Crop an image with the bounding box. It labels the cropped region "
        "with a bounding box and crops the region with some margins around "
        "the bounding box to help with contextual understanding of the region."
    )
    description_zh = (
        "使用边界框裁剪图像。它会标记裁剪区域并在边界框周围添加一些边距，"
        "以帮助理解区域的上下文。"
    )
    
    parameters = {
        "type": "object",
        "properties": {
            "image": {
                "type": "string",
                "description": "Path to the image to crop"
            },
            "bbox": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4,
                "description": (
                    "Bounding box as [left, top, right, bottom], where each value "
                    "is a float between 0 and 1 representing the percentage of "
                    "image width/height from the top left corner at [0, 0]"
                )
            }
        },
        "required": ["image", "bbox"]
    }
    
    def call(self, params: Union[str, Dict]) -> str:
        """Execute crop operation.
        
        Args:
            params: Parameters containing image path and bbox
            
        Returns:
            JSON string with cropped image data
        """
        # Validate and parse parameters
        params_dict = self.verify_json_format_args(params)
        
        image_path = params_dict["image"]
        bbox = params_dict["bbox"]
        
        # Parse bbox if string
        if isinstance(bbox, str):
            try:
                bbox = ast.literal_eval(bbox)
            except:
                return json.dumps({"error": "Invalid bbox format"})
        
        # Validate bbox values
        if not all(0 <= x <= 1.0 for x in bbox):
            return json.dumps({"error": "Bounding box coordinates must be between 0 and 1"})
        
        # Process image
        try:
            image = image_processing(image_path)
        except Exception as e:
            return json.dumps({"error": f"Error loading image: {str(e)}"})
        
        # Visualize bbox on image
        image = visualize_bbox(image, bbox)
        
        # Convert percentage bbox to pixel coordinates
        W, H = image.size
        pixel_bbox = [bbox[0] * W, bbox[1] * H, bbox[2] * W, bbox[3] * H]
        
        # Expand bbox with margins (default 50% margin)
        expanded_bbox = expand_bbox(pixel_bbox, image.size, margin=0.5)
        
        # Crop image
        cropped_image = image.crop(expanded_bbox)
        
        # Return image object in result
        result = {
            "image": cropped_image,
            "original_size": list(image.size),
            "cropped_size": list(cropped_image.size),
            "bbox": bbox,
            "expanded_bbox": list(expanded_bbox)
        }
        
        return json.dumps(result, default=str)

