"""Tool for estimating depth of an object in an image using DepthAnything model."""

import json
import numpy as np
from typing import Union, Dict

from tool.base_tool import BasicTool, register_tool
from tool.base_tool import TOOL_REGISTRY


@register_tool(name="estimate_object_depth")
class EstimateObjectDepthTool(BasicTool):
    """A tool to estimate the depth of an object in an image using DepthAnything model."""
    
    name = "estimate_object_depth"
    description_en = "Estimate the depth of an object in an image using DepthAnything model. It returns an estimated depth value of the object specified by a brief text description. The smaller the value is, the closer the object is to the camera, and the larger the farther. This tool may help you to better reason about the spatial relationship, like which object is closer to the camera."
    description_zh = "使用 DepthAnything 模型估计图像中对象的深度。返回由简短文本描述指定的对象的估计深度值。值越小，表示对象离相机越近；值越大，表示对象离相机越远。此工具可以帮助您更好地推理空间关系，例如哪个对象更接近相机。"
    
    parameters = {
        "type": "object",
        "properties": {
            "image": {
                "type": "string",
                "description": "The image to get the depth from"
            },
            "object": {
                "type": "string",
                "description": "A short description of the object to get the depth from"
            },
            "mode": {
                "type": "string",
                "enum": ["mean", "min", "max"],
                "description": "Mode to compute depth: mean (average), min (closest), or max (farthest)",
                "default": "mean"
            }
        },
        "required": ["image", "object"]
    }
    example = '{"image": "image-0", "object": "a black cat", "mode": "mean"}'
    
    def call(self, params: Union[str, Dict]) -> str:
        """Execute the object depth estimation operation.
        
        Args:
            params: Parameters containing the image path, object description, and mode
            
        Returns:
            JSON string with estimated depth value
        """
        # Validate and parse parameters
        params_dict = self.parse_params(params)
        image_path = params_dict["image"]
        object_desc = params_dict["object"]
        mode = params_dict.get("mode", "mean")
        
        try:
            # Use LocalizeObjects to find the object
            localize_tool_class = TOOL_REGISTRY.get("localize_objects")
            if localize_tool_class is None:
                return json.dumps({
                    "success": False,
                    "error": "localize_objects tool is not available"
                })
            
            localize_tool = localize_tool_class()
            localize_params = {
                "image": image_path,
                "objects": [object_desc]
            }
            localize_result = localize_tool.call(localize_params)
            
            # Parse the result
            if isinstance(localize_result, str):
                localize_data = json.loads(localize_result)
            else:
                localize_data = localize_result
            
            # Check if object was found
            if not localize_data.get("success", False) or len(localize_data.get("regions", [])) == 0:
                return json.dumps({
                    "depth": "Object not found."
                })
            
            # Use the best match object's bbox (highest score)
            regions = localize_data["regions"]
            best_match_idx = np.argmax([region.get("score", 0) for region in regions])
            bbox = regions[best_match_idx]["bbox"]
            
            # Use EstimateRegionDepth to estimate depth
            estimate_depth_tool_class = TOOL_REGISTRY.get("estimate_region_depth")
            if estimate_depth_tool_class is None:
                return json.dumps({
                    "success": False,
                    "error": "estimate_region_depth tool is not available"
                })
            
            estimate_depth_tool = estimate_depth_tool_class()
            depth_params = {
                "image": image_path,
                "bbox": bbox,
                "mode": mode
            }
            depth_result = estimate_depth_tool.call(depth_params)
            
            # Parse and return the depth result
            if isinstance(depth_result, str):
                depth_data = json.loads(depth_result)
            else:
                depth_data = depth_result
            
            if depth_data.get("success", False):
                return json.dumps({
                    "depth": depth_data.get("depth")
                })
            else:
                return json.dumps({
                    "success": False,
                    "error": depth_data.get("error", "Failed to estimate depth")
                })
                
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Error estimating object depth: {str(e)}"
            })

