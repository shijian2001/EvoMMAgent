from typing import Union, Dict
import os

from tool.base_tool import ModelBasedTool, register_tool
from tool.utils.image_utils import image_processing, get_full_path_data

OCR_PROMPT = "OCR:"
MAX_PIXELS_OCR = 1280 * 28 * 28


@register_tool(name="ocr")
class OCRTool(ModelBasedTool):
    """A tool to extract text from images using PaddleOCR-VL."""

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
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        from tool.model_config import PADDLE_OCR_VL_MODEL_PATH

        self.model = AutoModelForImageTextToText.from_pretrained(
            PADDLE_OCR_VL_MODEL_PATH, torch_dtype=torch.bfloat16
        ).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(PADDLE_OCR_VL_MODEL_PATH)
        self.device = device
        self.is_loaded = True

    def _call_impl(self, params: Union[str, Dict]) -> Dict:
        import torch
        from PIL import Image

        params_dict = self.parse_params(params)
        image_path = params_dict["image"]
        tmp_file_to_remove = None

        try:
            full_path = image_path if os.path.exists(image_path) else get_full_path_data(image_path)
            if full_path and os.path.exists(full_path):
                image_file = full_path
            else:
                image = image_processing(image_path)
                if isinstance(image, Image.Image):
                    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    tmp_file_to_remove = tmp.name
                    try:
                        image.save(tmp.name)
                        image_file = tmp.name
                    finally:
                        tmp.close()
                elif isinstance(image, str):
                    resolved = image if os.path.exists(image) else get_full_path_data(image)
                    if resolved is None:
                        raise FileNotFoundError(f"Image not found: {image_path}")
                    image_file = resolved
                else:
                    raise ValueError(f"Unexpected image type: {type(image)}")

            image = Image.open(image_file).convert("RGB")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": OCR_PROMPT},
                    ]
                }
            ]
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                images_kwargs={
                    "size": {
                        "shortest_edge": self.processor.image_processor.min_pixels,
                        "longest_edge": MAX_PIXELS_OCR,
                    }
                },
            )

            if "pixel_values" in inputs and not isinstance(inputs["pixel_values"], torch.Tensor):
                inputs["pixel_values"] = torch.from_numpy(inputs["pixel_values"]).to(self.device)
            if "image_grid_thw" in inputs and not isinstance(inputs["image_grid_thw"], torch.Tensor):
                inputs["image_grid_thw"] = torch.from_numpy(inputs["image_grid_thw"]).to(self.device)

            inputs = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

            outputs = self.model.generate(**inputs, max_new_tokens=512)
            extracted_text = self.processor.decode(
                outputs[0][inputs["input_ids"].shape[-1] : -1]
            ).strip()

            return {"extracted text": extracted_text}
        except FileNotFoundError as e:
            return {"error": f"Image not found: {str(e)}"}
        except Exception as e:
            import traceback
            return {"error": f"OCR error: {str(e)}\n{traceback.format_exc()}"}
        finally:
            if tmp_file_to_remove and os.path.exists(tmp_file_to_remove):
                try:
                    os.unlink(tmp_file_to_remove)
                except OSError:
                    pass