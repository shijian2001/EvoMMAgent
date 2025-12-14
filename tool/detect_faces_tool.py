"""Tool for detecting faces in images using DSFD face detection model."""

import json
import os
from typing import Union, Dict

from tool.base_tool import ModelBasedTool, register_tool
from tool.utils.image_utils import image_processing
from tool.utils.temp_manager import get_temp_manager
from tool.visualize_regions_tool import VisualizeRegionsOnImageTool


@register_tool(name="detect_faces")
class DetectFacesTool(ModelBasedTool):
    """A tool to detect faces in an image using DSFD face detection model."""
    
    name = "detect_faces"
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
    
    def load_model(self, device: str) -> None:
        """Load the DSFD face detection model to the specified device.
        
        Args:
            device: Device to load the model on (automatically selected by GPU manager)
        """
        import face_detection
        from tool.model_config import FACE_DETECTION_CHECKPOINT_PATH
        
        # Ensure checkpoint exists, download if needed
        if not os.path.exists(FACE_DETECTION_CHECKPOINT_PATH):
            os.makedirs(os.path.dirname(FACE_DETECTION_CHECKPOINT_PATH), exist_ok=True)
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id="zixianma/mma",
                filename="WIDERFace_DSFD_RES152.pth",
                local_dir=os.path.dirname(FACE_DETECTION_CHECKPOINT_PATH)
            )
        
        # Build detector
        self.model = face_detection.build_detector(
            "DSFDDetector",
            confidence_threshold=0.5,
            nms_iou_threshold=0.3
        )
        self.device = device
        self.is_loaded = True
    
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
                faces = self.model.detect(np.array(image))
            
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
            visualize_result = visualize_tool.call(visualize_params)
            
            # Extract output image path from visualize result
            if "Image saved to: " in visualize_result:
                output_image_path = visualize_result.replace("Image saved to: ", "")
            else:
                # Fallback: save image manually if visualization failed
                output_image_path = get_temp_manager().get_output_path(
                    "detect_faces", image_path, "faces_detected", ".png"
                )
                image.save(output_image_path)
            
            return json.dumps({
                "success": True,
                "image": output_image_path,
                "regions": regions
            })
            
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

