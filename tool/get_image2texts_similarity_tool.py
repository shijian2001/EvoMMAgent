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
    description_en = "Get the similarity between one image and a list of texts. Note that this similarity score may not be accurate and should be used as a reference only."
    description_zh = "计算一个图像与一组文本之间的相似度。注意此相似度分数可能不准确，仅作为参考。"
    
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
            },
            "tool_version": {
                "type": "string",
                "description": "CLIP model version (optional, defaults to ViT-H-14-378-quickgelu)",
                "default": "ViT-H-14-378-quickgelu"
            }
        },
        "required": ["image", "texts"]
    }
    example = '{"image": "image-0", "texts": ["a cat", "a dog"]}'
    
    def __init__(self, cfg=None, use_zh=False):
        """Initialize the GetImageToTextsSimilarity tool."""
        super().__init__(cfg, use_zh)
        self.model_version = None
        self.preprocess = None
        self.tokenizer = None
    
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
            pretrained=pretrained
        )
        self.model.eval()
        self.model = self.model.to(device)
        self.tokenizer = open_clip.get_tokenizer(model_version)
        self.device = device
        self.is_loaded = True
        self.model_version = model_version
    
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
        tool_version = params_dict.get("tool_version", "ViT-H-14-378-quickgelu")
        
        if not isinstance(texts, list) or len(texts) == 0:
            return json.dumps({
                "success": False,
                "error": "texts must be a non-empty list of strings"
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
            best_text = texts[best_text_index]
            
            return json.dumps({
                "success": True,
                "similarity": sim_scores,
                "best_text_index": best_text_index,
                "best_text": best_text
            })
            
        except FileNotFoundError as e:
            return json.dumps({
                "success": False,
                "error": f"Image file not found: {str(e)}"
            })
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Error computing image-to-text similarity: {str(e)}"
            })

