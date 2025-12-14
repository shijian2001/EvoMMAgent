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
    model_id = "clip"  # Automatic model sharing with other CLIP tools!
    
    description_en = "Get the similarity between one text and a list of images. Note that this similarity score may not be accurate and should be used as a reference only."
    description_zh = "计算一个文本与一组图像之间的相似度。注意此相似度分数可能不准确，仅作为参考。"
    
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
            },
            "tool_version": {
                "type": "string",
                "description": "CLIP model version (optional, defaults to ViT-H-14-378-quickgelu)",
                "default": "ViT-H-14-378-quickgelu"
            }
        },
        "required": ["text", "images"]
    }
    example = '{"text": "a black and white cat", "images": ["image-0", "image-1"]}'
    
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
            return json.dumps({
                "success": False,
                "error": "images must be a non-empty list of image paths"
            })
        
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
            best_image = images[best_image_index]
            
            return json.dumps({
                "success": True,
                "similarity": sim_scores,
                "best_image_index": best_image_index,
                "best_image": best_image
            })
            
        except FileNotFoundError as e:
            return json.dumps({
                "success": False,
                "error": f"Image file not found: {str(e)}"
            })
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Error computing text-to-image similarity: {str(e)}"
            })

