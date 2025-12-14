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
    description_en = "Get the similarity between one image and a list of other images. Note that this similarity score may not be accurate and should be used as a reference only."
    description_zh = "计算一个图像与一组其他图像之间的相似度。注意此相似度分数可能不准确，仅作为参考。"
    
    parameters = {
        "type": "object",
        "properties": {
            "image": {
                "type": "string",
                "description": "The reference image"
            },
            "other_images": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The other images to compare to the reference image"
            },
            "tool_version": {
                "type": "string",
                "description": "CLIP model version (optional, defaults to ViT-H-14-378-quickgelu)",
                "default": "ViT-H-14-378-quickgelu"
            }
        },
        "required": ["image", "other_images"]
    }
    example = '{"image": "image-0", "other_images": ["image-1", "image-2"]}'
    
    def __init__(self, cfg=None, use_zh=False):
        """Initialize the GetImageToImagesSimilarity tool."""
        super().__init__(cfg, use_zh)
        self.model_version = None
        self.preprocess = None
    
    def load_model(self, device: str) -> None:
        """Load the CLIP model to the specified device.
        
        Args:
            device: Device to load the model on (automatically selected by GPU manager)
        """
        import open_clip
        from tool.model_config import CLIP_MODEL_VERSION, CLIP_MODEL_PRETRAINED
        
        # Use default model version if not set
        model_version = self.model_version or CLIP_MODEL_VERSION
        pretrained = CLIP_MODEL_PRETRAINED
        
        # Load model, transforms, and tokenizer
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_version,
            pretrained=pretrained,
        )
        self.model.eval()
        self.model = self.model.to(device)
        self.device = device
        self.is_loaded = True
        # self.model_version = model_version
    
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
        tool_version = params_dict.get("tool_version", "ViT-H-14-378-quickgelu")
        
        if not isinstance(other_images, list) or len(other_images) == 0:
            return json.dumps({
                "success": False,
                "error": "other_images must be a non-empty list of image paths"
            })
        
        try:
            # Check if model needs to be reloaded (different version)
            if not self.is_loaded or self.model is None or tool_version != self.model_version:
                if self.is_loaded:
                    self.unload_model()
                # Set model version before loading
                self.model_version = tool_version
                self.load_model(self.device)
            
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
            best_image = other_images[best_image_index]
            
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
                "error": f"Error computing image similarity: {str(e)}"
            })

