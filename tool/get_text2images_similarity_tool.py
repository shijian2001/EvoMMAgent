"""Tool for computing similarity between one text and a list of images using CLIP."""

import json
import torch
from typing import Union, Dict, List

from tool.base_tool import ModelBasedTool, register_tool
from tool.utils.image_utils import image_processing


@register_tool(name="get_text2images_similarity")
class GetTextToImagesSimilarityTool(ModelBasedTool):
    """A tool to compute CLIP similarity between one text and a list of images."""
    
    name = "get_text2images_similarity"
    model_id = "clip"
    
    description_en = "Compute similarity between one text and multiple images (text → images)."
    description_zh = "计算一段文本与多张图像的相似度（文本 → 图像）。"
    
    parameters = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The reference text"
            },
            "images": {
                "type": "array",
                "items": {"type": "string"},
                "description": "A list of images to compare to the reference text"
            }
        },
        "required": ["text", "images"]
    }
    example = '{"text": "a black and white cat", "images": ["img_0", "img_1"]}'
    
    def load_model(self, device: str) -> None:
        import open_clip
        from tool.model_config import CLIP_VERSION, CLIP_PRETRAINED
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(CLIP_VERSION, pretrained=CLIP_PRETRAINED)
        self.model.eval()
        self.model = self.model.to(device)
        self.tokenizer = open_clip.get_tokenizer(CLIP_VERSION)
        self.device = device
        self.is_loaded = True
    
    def _call_impl(self, params: Union[str, Dict]) -> str:
        """Execute the text-to-image similarity computation operation.
        
        Args:
            params: Parameters containing the reference text, images, and optional model version
            
        Returns:
            JSON string with similarity scores, best image index, and best image
        """
        # Validate and parse parameters
        params_dict = self.parse_params(params)
        text = params_dict["text"]
        images = params_dict["images"]
        
        if not isinstance(images, list) or len(images) == 0:
            return {
                "error": "images must be a non-empty list of image paths"
            }
        
        try:
            # Tokenize reference text
            text_tokens = self.tokenizer([text]).to(self.device)
            
            # Process images
            images_processed = []
            for img_path in images:
                img = image_processing(img_path)
                images_processed.append(img)
            
            images_tensor = torch.stack([
                self.preprocess(img).to(self.device) for img in images_processed
            ])
            
            # Compute text and image features
            with torch.no_grad(), torch.cuda.amp.autocast():
                text_features = self.model.encode_text(text_tokens)
                image_features = self.model.encode_image(images_tensor)
                
                # Normalize features
                text_features /= text_features.norm(dim=-1, keepdim=True)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                # Compute similarity scores (cosine similarity)
                similarity_scores = text_features @ image_features.T
            
            # Convert to list and round (2 decimal places as in original code)
            sim_scores = [round(score.item(), 2) for score in similarity_scores[0]]
            
            # Find best match (argmax on the second dimension)
            best_image_index = torch.argmax(similarity_scores, dim=1).item()
            
            return {
                "similarity scores": sim_scores,
                "best match": images[best_image_index]
            }
            
        except FileNotFoundError as e:
            return {"error": f"Image file not found: {str(e)}"}
        except Exception as e:
            return {"error": f"Error computing text-to-image similarity: {str(e)}"}

