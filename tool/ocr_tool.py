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
    model_id = "ocr"
    
    description_en = "Extract texts from an image or return an empty string if no text is in the image."
    description_zh = "从图像中提取文本，如果图像中没有文本则返回空字符串。"
    
    parameters = {
        "type": "object",
        "properties": {
            "image": {
                "type": "string",
                "description": "Image ID (e.g., 'img_0')"
            }
        },
        "required": ["image"]
    }
    example = '{"image": "img_0"}'
    
    def load_model(self, device: str) -> None:
        import easyocr
        from tool.model_config import OCR_MODEL_DIR
        self.model = easyocr.Reader(["en"], gpu=device.startswith("cuda"), model_storage_directory=OCR_MODEL_DIR)
        self.device = device
        self.is_loaded = True
    
    def _call_impl(self, params: Union[str, Dict]) -> str:
        params_dict = self.parse_params(params)
        image_path = params_dict["image"]
        
        try:
            full_path = image_path if os.path.exists(image_path) else get_full_path_data(image_path)
            if full_path and os.path.exists(full_path):
                image_input = full_path
            else:
                image = image_processing(image_path)
                if isinstance(image, Image.Image):
                    image_input = np.array(image)
                elif isinstance(image, str):
                    image_input = image if os.path.exists(image) else get_full_path_data(image)
                    if image_input is None:
                        raise FileNotFoundError(f"Image not found: {image_path}")
                else:
                    raise ValueError(f"Unexpected image type: {type(image)}")
            
            result = self.model.readtext(image_input)
            result_text = [text for _, text, _ in result]
            extracted_text = ", ".join(result_text) if result_text else ""
            return {"extracted text": extracted_text}
        except FileNotFoundError as e:
            return {"error": f"Image not found: {str(e)}"}
        except Exception as e:
            return {"error": f"OCR error: {str(e)}"}

