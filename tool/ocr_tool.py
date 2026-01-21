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
    
    description_en = "Extract text from an image. Returns empty if no text found."
    description_zh = "提取图像中的文本，无文本则返回空。"
    
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
        import os
        from tool.model_config import OCR_MODEL_DIR
        
        if device.startswith("cuda"):
            # EasyOCR doesn't support specifying device directly, use environment variable
            device_id = device.split(":")[-1]
            old_env = os.environ.get('CUDA_VISIBLE_DEVICES')
            
            try:
                # Temporarily set visible devices to target GPU only
                os.environ['CUDA_VISIBLE_DEVICES'] = device_id
                
                # Now EasyOCR will load to the only visible GPU (which maps to our target device)
                self.model = easyocr.Reader(
                    ["en"], 
                    gpu=True, 
                    model_storage_directory=OCR_MODEL_DIR
                )
            finally:
                # Always restore environment variable
                if old_env is not None:
                    os.environ['CUDA_VISIBLE_DEVICES'] = old_env
                elif 'CUDA_VISIBLE_DEVICES' in os.environ:
                    del os.environ['CUDA_VISIBLE_DEVICES']
        else:
            # CPU mode
            self.model = easyocr.Reader(
                ["en"], 
                gpu=False, 
                model_storage_directory=OCR_MODEL_DIR
            )
        
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

