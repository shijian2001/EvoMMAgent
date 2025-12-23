"""Tool for viewing generated images from memory."""

from typing import Union, Dict
from tool.base_tool import BasicTool, register_tool


@register_tool(name="get_images")
class GetImagesTool(BasicTool):
    """View one or multiple generated images from memory by their IDs."""
    
    name = "get_images"
    description_en = "View generated images by their IDs to verify tool outputs or understand image content. Use this when you need to visually inspect intermediate results."
    description_zh = "通过 ID 查看生成的图片，用于验证工具输出或理解图片内容。当你需要视觉检查中间结果时使用此工具。"
    
    parameters = {
        "type": "object",
        "properties": {
            "images": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of image IDs to view (e.g., ['img_1', 'img_2'])"
            }
        },
        "required": ["images"]
    }
    example = '{"images": ["img_1", "img_2"]}'
    
    def call(self, params: Union[str, Dict], **kwargs) -> Dict:
        """Execute the image viewing operation.
        
        Args:
            params: Parameters containing image IDs to view
            **kwargs: Additional arguments (not used)
            
        Returns:
            Dict with special marker for agent processing
        """
        # Parse parameters
        params_dict = self.parse_params(params)
        image_ids = params_dict["images"]
        
        # Ensure it's a list
        if not isinstance(image_ids, list):
            image_ids = [image_ids]
        
        # Validate image IDs format only
        for img_id in image_ids:
            if not isinstance(img_id, str):
                return {
                    "error": f"Invalid image ID type: {type(img_id).__name__}. Expected string."
                }
            if not (img_id.startswith("img_") or img_id.startswith("vid_")):
                return {
                    "error": f"Invalid image ID format: {img_id}. Expected format: 'img_0', 'img_1', etc."
                }
        
        # Generate concise observation message
        count = len(image_ids)
        if count == 1:
            message = f"You can now view {image_ids[0]}."
        else:
            # Format: "img_1, img_2 and img_3"
            ids_list = ", ".join(image_ids[:-1]) + f" and {image_ids[-1]}"
            message = f"You can now view {count} images in order: {ids_list}."
        
        return {
            "_get_images_ids": image_ids,  # Special marker for agent
            "message": message             # Observation text for LLM
        }
