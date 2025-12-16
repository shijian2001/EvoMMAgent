"""Tool integration test."""

import asyncio
import json
import os
from pathlib import Path
from tool import TOOL_REGISTRY

print(f"Registered tools: {list(TOOL_REGISTRY.keys())}\n")

# Optional: preload models to avoid first-call latency
# from tool.model_config import preload_tools
# preload_tools(tool_bank=["estimate_region_depth", "estimate_object_depth"])

IMG = "test_image.png"
# IMG2 = "test_image2.png"
# IMG3 = "test_image3.png"

# Output directory for multimodal results
OUTPUT_DIR = Path("test_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def save_multimodal_output(result, tool_name: str):
    """Save multimodal outputs (images, videos) to disk.
    
    Args:
        result: Tool result (dict or JSON string)
        tool_name: Name of the tool for organizing outputs
    """
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except:
            return result
    
    if not isinstance(result, dict):
        return result
    
    # Check for errors
    if "error" in result:
        print(f"  ❌ Error: {result['error']}")
        return result
    
    # Handle PIL Image output
    if "output_image" in result:
        try:
            from PIL import Image
            img = result["output_image"]
            if isinstance(img, Image.Image):
                output_path = OUTPUT_DIR / f"{tool_name}_output.png"
                img.save(output_path)
                result["output_image"] = str(output_path)
                print(f"  💾 Saved image to: {output_path}")
        except Exception as e:
            print(f"  ⚠️  Failed to save image: {e}")
    
    # Handle video output
    if "output_video" in result:
        try:
            import shutil
            video_path = result["output_video"]
            if isinstance(video_path, str) and os.path.exists(video_path):
                ext = os.path.splitext(video_path)[1]
                output_path = OUTPUT_DIR / f"{tool_name}_output{ext}"
                shutil.copy2(video_path, output_path)
                result["output_video"] = str(output_path)
                print(f"  💾 Saved video to: {output_path}")
        except Exception as e:
            print(f"  ⚠️  Failed to save video: {e}")
    
    return result


async def test():
    # calculator
    # r = await TOOL_REGISTRY["calculator"]().call_async({"expression": "2 + 3 * 4"})
    # print(f"calculator: {r}")
    
    # crop
    # r = await TOOL_REGISTRY["crop"]().call_async({"image": IMG, "bbox": [0.1, 0.2, 0.5, 0.6]})
    # r = save_multimodal_output(r, "crop")
    # print(f"crop: {r}")
    
    # zoom_in
    # r = await TOOL_REGISTRY["zoom_in"]().call_async({"image": IMG, "bbox": [0.2, 0.2, 0.8, 0.8], "zoom_factor": 2.0})
    # r = save_multimodal_output(r, "zoom_in")
    # print(f"zoom_in: {r}")
    
    # visualize_regions
    # r = await TOOL_REGISTRY["visualize_regions"]().call_async({
    #     "image_path": IMG, 
    #     "regions": [{"bbox": [0.1, 0.1, 0.5, 0.5], "label": "A"}, {"bbox": [0.5, 0.5, 0.9, 0.9], "label": "B"}]
    # })
    # r = save_multimodal_output(r, "visualize_regions")
    # print(f"visualize_regions: {r}")
    
    # solve_math_equation (skip if no API key)
    # r = await TOOL_REGISTRY["solve_math_equation"]().call_async({"query": "solve x^2 - 4 = 0"})
    # print(f"solve_math_equation: {r}")

    # get_objects
    # r = await TOOL_REGISTRY["get_objects"]().call_async({"image": IMG})
    # print(f"get_objects: {r}")

    # localize_objects
    # r = await TOOL_REGISTRY["localize_objects"]().call_async({"image": IMG, "objects": ["bread", "orange"]})
    # r = save_multimodal_output(r, "localize_objects")
    # print(f"localize_objects: {r}")

    # detect_faces
    # r = await TOOL_REGISTRY["detect_faces"]().call_async({"image": IMG})
    # r = save_multimodal_output(r, "detect_faces")
    # print(f"detect_faces: {r}")

    # estimate_region_depth
    r = await TOOL_REGISTRY["estimate_region_depth"]().call_async({"image": IMG, "bbox": [0.1, 0.2, 0.5, 0.6], "mode": "mean"})
    print(f"estimate_region_depth: {r}")

    # estimate_object_depth
    r = await TOOL_REGISTRY["estimate_object_depth"]().call_async({"image": IMG, "object": "nut"})
    print(f"estimate_object_depth: {r}")

    # get_image2images_similarity
    # r = await TOOL_REGISTRY["get_image2images_similarity"]().call_async({"image": IMG, "other_images": [IMG2]})
    # print(f"get_image2images_similarity: {r}")

    # get_image2texts_similarity
    # r = await TOOL_REGISTRY["get_image2texts_similarity"]().call_async({"image": IMG, "texts": ["a photo of a dog"]})
    # print(f"get_image2texts_similarity: {r}")

    # get_text2images_similarity
    # r = await TOOL_REGISTRY["get_text2images_similarity"]().call_async({"text": "a photo of a food", "images": [IMG]})
    # print(f"get_text2images_similarity: {r}")

    # ocr
    # r = await TOOL_REGISTRY["ocr"]().call_async({"image": IMG3})
    # print(f"ocr: {r}")

if __name__ == "__main__":
    asyncio.run(test())
