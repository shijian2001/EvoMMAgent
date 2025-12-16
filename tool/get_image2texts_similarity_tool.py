"""Tool for computing similarity between one image and a list of texts using CLIP."""

import json
import torch
from typing import Union, Dict, List

from tool.base_tool import ModelBasedTool, register_tool
from tool.utils.image_utils import image_processing


@register_tool(name="get_image2texts_similarity")
class GetImageToTextsSimilarityTool(ModelBasedTool):
    """A tool to compute CLIP similarity between one image and a list of texts."""
    
    name = "get_image2texts_similarity"
    model_id = "clip"
    
    description_en = "Get the similarity between one image and a list of texts."
    description_zh = "计算一个图像与一组文本之间的相似度。"
    
    parameters = {
        "type": "object",
        "properties": {
            "image": {
                "type": "string",
                "description": "The reference image"
            },
            "texts": {
                "type": "array",
                "items": {"type": "string"},
                "description": "A list of texts to compare to the reference image"
            }
        },
        "required": ["image", "texts"]
    }
    example = '{"image": "img_0", "texts": ["a cat", "a dog"]}'
    
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
        """Execute the image-to-text similarity computation operation.
        
        Args:
            params: Parameters containing the reference image, texts, and optional model version
            
        Returns:
            JSON string with similarity scores, best text index, and best text
        """
        # Validate and parse parameters
        params_dict = self.parse_params(params)
        image_path = params_dict["image"]
        texts = params_dict["texts"]
        
        if not isinstance(texts, list) or len(texts) == 0:
            return {
                "error": "texts must be a non-empty list of strings"
            }
        
        try:
            # Process reference image
            reference_image = image_processing(image_path)
            reference_image_tensor = self.preprocess(reference_image).unsqueeze(0).to(self.device)
            
            # Tokenize texts
            text_tokens = self.tokenizer(texts).to(self.device)
            
            # Compute image and text features
            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = self.model.encode_image(reference_image_tensor)
                text_features = self.model.encode_text(text_tokens)
                
                # Normalize features
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                # Compute similarity scores (cosine similarity)
                similarity_scores = image_features @ text_features.T
            
            # Convert to list and round (2 decimal places as in original code)
            sim_scores = [round(score.item(), 2) for score in similarity_scores[0]]
            
            # Find best match (argmax on the second dimension)
            best_text_index = torch.argmax(similarity_scores, dim=1).item()
            
            return {
                "similarity scores": sim_scores,
                "best match": texts[best_text_index]
            }
            
        except FileNotFoundError as e:
            return {"error": f"Image file not found: {str(e)}"}
        except Exception as e:
            return {"error": f"Error computing image-to-text similarity: {str(e)}"}

