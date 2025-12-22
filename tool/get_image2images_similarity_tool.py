"""Tool for computing similarity between one image and a list of other images using CLIP."""

import json
import torch
from typing import Union, Dict, List

from tool.base_tool import ModelBasedTool, register_tool
from tool.utils.image_utils import image_processing


@register_tool(name="get_image2images_similarity")
class GetImageToImagesSimilarityTool(ModelBasedTool):
    """A tool to compute CLIP similarity between one image and a list of other images."""
    
    name = "get_image2images_similarity"
    model_id = "clip"
    
    description_en = "Get the similarity between one image and a list of other images."
    description_zh = "计算一个图像与一组其他图像之间的相似度。"
    
    parameters = {
        "type": "object",
        "properties": {
            "image": {
                "type": "string",
                "description": "Image ID (e.g., 'img_0')"
            },
            "other_images": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The other images to compare to the reference image"
            }
        },
        "required": ["image", "other_images"]
    }
    example = '{"image": "img_0", "other_images": ["img_1", "img_2"]}'
    
    def load_model(self, device: str) -> None:
        import open_clip
        from tool.model_config import CLIP_VERSION, CLIP_PRETRAINED
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(CLIP_VERSION, pretrained=CLIP_PRETRAINED)
        self.model.eval()
        self.model = self.model.to(device)
        self.device = device
        self.is_loaded = True
    
    def _call_impl(self, params: Union[str, Dict]) -> str:
        """Execute the image similarity computation operation.
        
        Args:
            params: Parameters containing the reference image, other images, and optional model version
            
        Returns:
            JSON string with similarity scores, best image index, and best image
        """
        # Validate and parse parameters
        params_dict = self.parse_params(params)
        image_path = params_dict["image"]
        other_images = params_dict["other_images"]
        
        if not isinstance(other_images, list) or len(other_images) == 0:
            return {
                "error": "other_images must be a non-empty list of image paths"
            }
        
        try:
            # Process reference image
            reference_image = image_processing(image_path)
            reference_image_tensor = self.preprocess(reference_image).unsqueeze(0).to(self.device)
            
            # Process other images
            other_images_processed = []
            for img_path in other_images:
                img = image_processing(img_path)
                other_images_processed.append(img)
            
            other_images_tensor = torch.stack([
                self.preprocess(img).to(self.device) for img in other_images_processed
            ])
            
            # Compute image features
            with torch.no_grad(), torch.cuda.amp.autocast():
                reference_features = self.model.encode_image(reference_image_tensor)
                other_features = self.model.encode_image(other_images_tensor)
                
                # Normalize features
                reference_features /= reference_features.norm(dim=-1, keepdim=True)
                other_features /= other_features.norm(dim=-1, keepdim=True)
                
                # Compute similarity scores (cosine similarity)
                similarity_scores = reference_features @ other_features.T
            
            # Convert to list and round (2 decimal places as in original code)
            sim_scores = [round(score.item(), 2) for score in similarity_scores[0]]
            
            # Find best match (argmax on the second dimension)
            best_image_index = torch.argmax(similarity_scores, dim=1).item()
            
            return {
                "similarity scores": sim_scores,
                "best match": other_images[best_image_index]
            }
            
        except FileNotFoundError as e:
            return {"error": f"Image file not found: {str(e)}"}
        except Exception as e:
            return {"error": f"Error computing image similarity: {str(e)}"}

