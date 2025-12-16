"""Tool for estimating depth of a region in an image using DepthAnything model."""

import json
import numpy as np
import torch
from typing import Union, Dict

from tool.base_tool import ModelBasedTool, register_tool
from tool.utils.image_utils import image_processing


@register_tool(name="estimate_region_depth")
class EstimateRegionDepthTool(ModelBasedTool):
    """A tool to estimate the depth of a specific region in an image using DepthAnything model."""
    
    name = "estimate_region_depth"
    model_id = "depth_anything"
    
    description_en = "Estimate the depth of a specific region in an image using DepthAnything model. The smaller the value is, the closer the region is to the camera."
    description_zh = "估计图像中特定区域的深度。值越小，表示该区域离相机越近。"
    
    parameters = {
        "type": "object",
        "properties": {
            "image": {
                "type": "string",
                "description": "Path to the image"
            },
            "bbox": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4,
                "description": "Bounding box [x1, y1, x2, y2] in normalized coordinates (0-1)"
            },
            "mode": {
                "type": "string",
                "enum": ["mean", "min", "max"],
                "description": "Mode to compute depth: mean (average), min (closest), or max (farthest)",
                "default": "mean"
            }
        },
        "required": ["image", "bbox"]
    }
    example = '{"image": "img_0", "bbox": [0.1, 0.2, 0.5, 0.6], "mode": "mean"}'
    
    def load_model(self, device: str) -> None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        import torch
        import os
        from tool.model_config import DEPTH_ANYTHING_PATH
        
        model_id = DEPTH_ANYTHING_PATH
        self.image_processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.is_loaded = True
    
    def _call_impl(self, params: Union[str, Dict]) -> str:
        """Execute the depth estimation operation.
        
        Args:
            params: Parameters containing the image path, bbox, and mode
            
        Returns:
            JSON string with estimated depth value
        """
        # Validate and parse parameters
        params_dict = self.parse_params(params)
        image_path = params_dict["image"]
        bbox = params_dict["bbox"]
        mode = params_dict.get("mode", "mean")
        
        # Validate bbox
        if not all(0 <= x <= 1.0 for x in bbox):
            return {"error": "bbox values must be between 0 and 1"}
        if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            return {"error": "invalid bbox: left >= right or top >= bottom"}
        
        try:
            from PIL import Image

            # Load and process image
            image = image_processing(image_path)
            W, H = image.size

            # Convert normalized bbox to pixel coordinates
            x1 = int(bbox[0] * W)
            y1 = int(bbox[1] * H)
            x2 = int(bbox[2] * W)
            y2 = int(bbox[3] * H)

            inputs = self.image_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            processed_outputs = self.image_processor.post_process_depth_estimation(
                outputs,
                target_sizes=[(H, W)],
            )
            depth_map = processed_outputs[0]["predicted_depth"].squeeze().detach().cpu().numpy()

            # Extract depth values from the region
            region_depth = depth_map[y1:y2, x1:x2]
            
            # Compute depth based on mode
            if mode == "mean":
                depth_value = float(np.mean(region_depth))
            elif mode == "min":
                depth_value = float(np.min(region_depth))
            elif mode == "max":
                depth_value = float(np.max(region_depth))
            else:
                depth_value = float(np.mean(region_depth))
            
            return {"estimated depth": round(depth_value, 4)}
            
        except FileNotFoundError as e:
            return {"error": f"Image file not found: {str(e)}"}
        except Exception as e:
            return {"error": f"Error estimating depth: {str(e)}"}

