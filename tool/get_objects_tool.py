"""Tool for detecting objects in images using RAM (Recognize Anything Model)."""

import json
from typing import Union, Dict

from tool.base_tool import ModelBasedTool, register_tool
from tool.utils.image_utils import image_processing


@register_tool(name="get_objects")
class GetObjectsTool(ModelBasedTool):
    """A tool to detect objects in an image using RAM model."""
    
    name = "get_objects"
    model_id = "ram"
    
    description_en = "Detect and extract objects from an image using the Recognize Anything Model (RAM)."
    description_zh = "从图像中检测和提取对象。"
    
    parameters = {
        "type": "object",
        "properties": {
            "image": {
                "type": "string",
                "description": "Path to the image to get objects from"
            }
        },
        "required": ["image"]
    }
    example = '{"image": "img_0"}'
    
    def load_model(self, device: str) -> None:
        from ram.models import ram_plus
        from tool.model_config import RAM_MODEL_PATH
        self.model = ram_plus(pretrained=RAM_MODEL_PATH, image_size=384, vit="swin_l")
        self.model.eval()
        self.model = self.model.to(device)
        self.device = device
        self.is_loaded = True
    
    def _call_impl(self, params: Union[str, Dict]) -> str:
        """Execute the object detection operation.
        
        Args:
            params: Parameters containing the image path
            
        Returns:
            JSON string with detected objects
        """
        # Validate and parse parameters
        params_dict = self.parse_params(params)
        image_path = params_dict["image"]
        
        try:
            from ram import get_transform, inference_ram_openset as inference
            
            # Process image
            image_size = 384
            transform = get_transform(image_size=image_size)
            image = image_processing(image_path)
            image = transform(image).unsqueeze(0).to(self.device)
            
            # Run inference
            tags = inference(image, self.model)
            objs = tags.split(" | ")
            
            return {"detected objects": objs}
            
        except FileNotFoundError as e:
            return {"error": f"Image file not found: {str(e)}"}
        except Exception as e:
            return {"error": f"Error detecting objects: {str(e)}"}

