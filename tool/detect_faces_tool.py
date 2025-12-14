"""Tool for detecting faces in images using DSFD face detection model."""

import json
import os
from typing import Union, Dict

from tool.base_tool import ModelBasedTool, register_tool
from tool.utils.image_utils import image_processing
from tool.visualize_regions_tool import VisualizeRegionsOnImageTool


@register_tool(name="detect_faces")
class DetectFacesTool(ModelBasedTool):
    """A tool to detect faces in an image using DSFD face detection model."""
    
    name = "detect_faces"
    model_id = "face_detection"  # Automatic model sharing
    
    description_en = "Detect faces in an image and return bounding boxes for each detected face."
    description_zh = "检测图像中的人脸，并返回每个人脸的边界框。"
    
    parameters = {
        "type": "object",
        "properties": {
            "image": {
                "type": "string",
                "description": "The image to detect faces from"
            }
        },
        "required": ["image"]
    }
    example = '{"image": "image-0"}'
    
    def enlarge_face(self, box, W, H, f=1.5):
        """Enlarge face bounding box by a factor.
        
        Args:
            box: Bounding box [x1, y1, x2, y2]
            W: Image width
            H: Image height
            f: Enlargement factor (default: 1.5)
            
        Returns:
            Enlarged bounding box [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = box
        w = int((f - 1) * (x2 - x1) / 2)
        h = int((f - 1) * (y2 - y1) / 2)
        x1 = max(0, x1 - w)
        y1 = max(0, y1 - h)
        x2 = min(W, x2 + w)
        y2 = min(H, y2 + h)
        return [x1, y1, x2, y2]
    
    def _call_impl(self, params: Union[str, Dict]) -> str:
        """Execute the face detection operation.
        
        Args:
            params: Parameters containing the image path
            
        Returns:
            JSON string with detected faces and visualized image
        """
        # Validate and parse parameters
        params_dict = self.parse_params(params)
        image_path = params_dict["image"]
        
        try:
            import numpy as np
            import torch
            
            # Load and process image
            image_path_full = image_processing(image_path, return_path=True)
            image = image_processing(image_path)
            W, H = image.size
            
            # Detect faces
            with torch.no_grad():
                faces = self.detector.detect(np.array(image))
            
            # Process results
            regions = []
            for i, box in enumerate(faces):
                x1, y1, x2, y2, c = [int(v) for v in box.tolist()]
                
                # Normalize bbox to [0, 1] range
                normalized_bbox = [
                    round(x1 / W, 4),
                    round(y1 / H, 4),
                    round(x2 / W, 4),
                    round(y2 / H, 4)
                ]
                
                # Create label (face 1, face 2, etc.)
                label = f'face {i+1}' if i > 0 else 'face'
                
                regions.append({
                    "bbox": normalized_bbox,
                    "label": label
                })
            
            # Visualize regions on image
            visualize_tool = VisualizeRegionsOnImageTool()
            visualize_params = {
                "image_path": image_path_full,
                "regions": regions
            }
            output_image = visualize_tool.call(visualize_params)
            
            # Return dict with PIL Image
            return {
                "success": True,
                "output_image": output_image,  # PIL.Image object
                "regions": regions
            }
            
        except FileNotFoundError as e:
            return json.dumps({
                "success": False,
                "error": f"Image file not found: {str(e)}"
            })
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Error detecting faces: {str(e)}"
            })

