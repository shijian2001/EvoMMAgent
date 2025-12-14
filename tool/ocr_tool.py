"""Tool for extracting text from images using EasyOCR."""

import json
import os
import io
import numpy as np
from PIL import Image
from typing import Union, Dict

from tool.base_tool import ModelBasedTool, register_tool
from tool.utils.image_utils import image_processing, get_full_path_data


@register_tool(name="ocr")
class OCRTool(ModelBasedTool):
    """A tool to extract text from images using EasyOCR."""
    
    name = "ocr"
    model_id = "ocr"  # Automatic model sharing across OCR tool instances
    
    description_en = "Extract texts from an image or return an empty string if no text is in the image. Note that the texts extracted may be incorrect or in the wrong order. It should be used as a reference only."
    description_zh = "从图像中提取文本，如果图像中没有文本则返回空字符串。注意提取的文本可能不正确或顺序错误，仅作为参考。"
    
    parameters = {
        "type": "object",
        "properties": {
            "image": {
                "type": "string",
                "description": "The image to extract texts from"
            }
        },
        "required": ["image"]
    }
    example = '{"image": "image-0"}'
    
    def _call_impl(self, params: Union[str, Dict]) -> str:
        """Execute the OCR operation.
        
        Args:
            params: Parameters containing the image path
            
        Returns:
            JSON string with extracted text
        """
        # Validate and parse parameters
        params_dict = self.parse_params(params)
        image_path = params_dict["image"]
        
        try:
            # Process image - get the original path if possible, otherwise process to PIL Image
            # First try to get the full path directly
            full_path = image_path if os.path.exists(image_path) else get_full_path_data(image_path)
            
            if full_path and os.path.exists(full_path):
                # Use file path directly (most efficient for EasyOCR)
                image_input = full_path
            else:
                # Process image to PIL Image and convert to numpy array
                image = image_processing(image_path)
                if isinstance(image, Image.Image):
                    # Convert PIL Image to numpy array (EasyOCR accepts numpy arrays)
                    image_input = np.array(image)
                elif isinstance(image, str):
                    # If it's a string path, use it directly
                    image_input = image if os.path.exists(image) else get_full_path_data(image)
                    if image_input is None:
                        raise FileNotFoundError(f"Image file not found: {image_path}")
                else:
                    raise ValueError(f"Unexpected image type: {type(image)}")
            
            # Run OCR
            result = self.reader.readtext(image_input)
            result_text = [text for _, text, _ in result]
            extracted_text = ", ".join(result_text) if result_text else ""
            
            return json.dumps({
                "success": True,
                "text": extracted_text
            })
                
        except FileNotFoundError as e:
            return json.dumps({
                "success": False,
                "error": f"Image file not found: {str(e)}"
            })
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Error extracting text: {str(e)}"
            })

