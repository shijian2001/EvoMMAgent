"""Multimodal content processing utilities for vision-language models.

Handles video and image inputs, converting them to API-compatible formats.

Environment Variables:
    FORCE_QWENVL_VIDEO_READER: Set video reader backend ('torchvision', 'decord', or 'torchcodec')
    
Notes:
    - image_patch_size: 14 for Qwen2.5-VL, 16 for Qwen3-VL
    - return_video_metadata: Only for Qwen3-VL (True), False for Qwen2.5-VL
"""

import base64
from io import BytesIO
from typing import Dict, Any, List, Tuple, Optional

from PIL import Image
from qwen_vl_utils import process_vision_info


def detect_qwen_version(model_name: str) -> Tuple[int, bool]:
    """Detect Qwen version from model name and return appropriate parameters.
    
    Args:
        model_name: Model name (e.g., "qwen2.5-vl-72b", "qwen3-vl-235b")
        
    Returns:
        Tuple of (image_patch_size, return_video_metadata)
    """
    model_name_lower = model_name.lower()
    
    if "qwen3" in model_name_lower or "qwen-3" in model_name_lower:
        return 16, True
    else:
        return 14, False


def build_video_message(
        user_prompt: List[Dict[str, Any]],
        image_patch_size: int = 14,
        return_video_metadata: bool = False
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Process user prompt containing video inputs and convert to API format.
    
    Args:
        user_prompt: List of content items with type and data
        image_patch_size: Patch size for vision processing (14 for Qwen2.5-VL, 16 for Qwen3-VL)
        return_video_metadata: Whether to return video metadata (Qwen3-VL only)
        
    Returns:
        Tuple of (processed_content, video_kwargs)
    """
    processed_content = []
    video_kwargs = {}
    
    for item in user_prompt:
        if item["type"] == "video":
            video_params = {k: v for k, v in item.items() if k != "type"}
            
            video_msg = [{
                "content": [{
                    "type": "video",
                    **video_params
                }]
            }]
            
            _, videos, video_kwargs = process_vision_info(
                video_msg,
                image_patch_size=image_patch_size,
                return_video_kwargs=True,
                return_video_metadata=return_video_metadata
            )
            
            # Split videos and metadata if return_video_metadata=True
            if videos is not None:
                if return_video_metadata and isinstance(videos[0], tuple):
                    video_tensors, video_metadatas = zip(*videos)
                    video_tensor = list(video_tensors)[0]
                else:
                    video_tensor = videos[0]
            else:
                raise ValueError("No video data returned from process_vision_info")
            
            frames = video_tensor.permute(0, 2, 3, 1).numpy().astype('uint8')
            
            base64_frames = [
                base64.b64encode(
                    (buffer := BytesIO(), 
                     Image.fromarray(frame).save(buffer, format="JPEG"), 
                     buffer.getvalue())[2]
                ).decode("utf-8")
                for frame in frames
            ]
            
            processed_content.append({
                "type": "video_url", 
                "video_url": {"url": f"data:video/jpeg;base64,{','.join(base64_frames)}"}
            })
        else:
            processed_content.append(item)
    
    return processed_content, video_kwargs


def build_image_message(
        user_prompt: List[Dict[str, Any]],
        image_patch_size: int = 14
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Process user prompt containing image inputs and convert to API format.
    
    Args:
        user_prompt: List of content items with type and data.
                    Image items should have "image" field with local file path or URL.
        image_patch_size: Patch size for vision processing (14 for Qwen2.5-VL, 16 for Qwen3-VL)
        
    Returns:
        Tuple of (processed_content, image_kwargs)
    """
    processed_content = []
    image_kwargs = {}
    
    for item in user_prompt:
        if item["type"] == "image":
            image_params = {k: v for k, v in item.items() if k != "type"}
            
            # Add default max_pixels if not specified
            if 'max_pixels' not in image_params:
                image_params['max_pixels'] = 28 * 28 * 128
            
            image_msg = [{
                "content": [{
                    "type": "image",
                    **image_params
                }]
            }]
            
            images, _ = process_vision_info(image_msg, image_patch_size=image_patch_size)
            
            if images:
                for img in images:
                    buffer = BytesIO()
                    img.save(buffer, format="JPEG")
                    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    
                    processed_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                    })
            
            param_keys = ['min_pixels', 'max_pixels', 'resized_height', 'resized_width']
            for key in param_keys:
                if key in image_params:
                    image_kwargs[key] = image_params[key]
        else:
            processed_content.append(item)
    
    return processed_content, image_kwargs


def build_multimodal_message(
        user_prompt: List[Dict[str, Any]],
        image_patch_size: int = 14,
        return_video_metadata: bool = False,
        model_name: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Route multimodal message processing based on content type.
    
    Args:
        user_prompt: List of content items with type and data
        image_patch_size: Patch size for vision processing (14 for Qwen2.5-VL, 16 for Qwen3-VL)
        return_video_metadata: Whether to return video metadata (Qwen3-VL only)
        model_name: Optional model name for auto-detection. If provided, overrides 
                   image_patch_size and return_video_metadata
        
    Returns:
        Tuple of (processed_content, mm_kwargs)
    """
    # Auto-detect parameters from model name if provided
    if model_name:
        image_patch_size, return_video_metadata = detect_qwen_version(model_name)
    
    content_types = {item.get("type") for item in user_prompt}
    
    if "video" in content_types:
        return build_video_message(user_prompt, image_patch_size, return_video_metadata)
    elif "image" in content_types or "image_url" in content_types:
        return build_image_message(user_prompt, image_patch_size)
    else:
        return user_prompt, {}
