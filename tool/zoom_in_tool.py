"""Zoom in tool for image region magnification."""

import ast
import json
from typing import Union, Dict
from PIL import Image

from tool.base_tool import BasicTool, register_tool
from tool.utils.image_utils import image_processing, expand_bbox, visualize_bbox
from tool.utils.temp_manager import get_temp_manager


@register_tool(name="zoom_in")
class ZoomInTool(BasicTool):
    """Zoom in on a region of an image.
    
    First crops the specified region from the image with the bounding box,
    then resizes the cropped region to create the zoom effect. Adds margins
    around the cropped region for contextual understanding.
    """
    
    name = "zoom_in"
    description_en = (
        "Zoom in on a region of the input image. This tool first crops the "
        "specified region from the image with the bounding box and then resizes "
        "the cropped region to create the zoom effect. It also adds some margins "
        "around the cropped region to help with contextual understanding."
    )
    description_zh = (
        "放大图像的某个区域。此工具首先使用边界框从图像中裁剪指定区域，"
        "然后调整裁剪区域的大小以创建缩放效果。它还会在裁剪区域周围添加一些边距，"
        "以帮助理解上下文。"
    )
    
    parameters = {
        "type": "object",
        "properties": {
            "image": {
                "type": "string",
                "description": "Path to the image to zoom in on"
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
            },
            "zoom_factor": {
                "type": "number",
                "description": "The factor to zoom in by (must be greater than 1)",
                "minimum": 1.0,
                "exclusiveMinimum": True
            }
        },
        "required": ["image", "bbox", "zoom_factor"]
    }
    
    def call(self, params: Union[str, Dict]) -> str:
        """Execute zoom in operation.
        
        Args:
            params: Parameters containing image path, bbox, and zoom_factor
            
        Returns:
            JSON string with zoomed image data
        """
        # Validate and parse parameters
        params_dict = self.verify_json_format_args(params)
        
        image_path = params_dict["image"]
        bbox = params_dict["bbox"]
        zoom_factor = params_dict["zoom_factor"]
        
        # Parse bbox if string
        if isinstance(bbox, str):
            try:
                bbox = ast.literal_eval(bbox)
            except:
                return json.dumps({"error": "Invalid bbox format"})
        
        # Validate zoom factor
        if zoom_factor <= 1:
            return json.dumps({"error": "Zoom factor must be greater than 1 to zoom in"})
        
        # Validate bbox values
        if not all(0 <= x <= 1.0 for x in bbox):
            return json.dumps({"error": "Bounding box coordinates must be between 0 and 1"})
        
        # Process image
        try:
            image = image_processing(image_path)
        except Exception as e:
            return json.dumps({"error": f"Error loading image: {str(e)}"})
        
        # Visualize and crop
        image = visualize_bbox(image, bbox)
        W, H = image.size
        pixel_bbox = [bbox[0] * W, bbox[1] * H, bbox[2] * W, bbox[3] * H]
        expanded_bbox = expand_bbox(pixel_bbox, image.size, margin=0.5)
        cropped_image = image.crop(expanded_bbox)
        
        # Calculate zoomed size
        crop_w, crop_h = cropped_image.size
        new_width = int(crop_w * zoom_factor)
        new_height = int(crop_h * zoom_factor)
        
        # Resize to create zoom effect
        zoomed_image = cropped_image.resize(
            (new_width, new_height),
            Image.LANCZOS
        )
        
        # Save zoomed image using temp manager
        temp_manager = get_temp_manager()
        output_path = temp_manager.get_output_path(
            tool_name="zoom_in",
            input_path=image_path if isinstance(image_path, str) else None,
            suffix="zoomed"
        )
        zoomed_image.save(output_path)
        
        # Return metadata with saved image path
        result = {
            "success": True,
            "output_image": output_path,
            "original_size": list(image.size),
            "cropped_size": [crop_w, crop_h],
            "zoomed_size": [new_width, new_height],
            "zoom_factor": zoom_factor,
            "bbox": bbox,
            "expanded_bbox": list(expanded_bbox)
        }
        
        return json.dumps(result, ensure_ascii=False, indent=2)

