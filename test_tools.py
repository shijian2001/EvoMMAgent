"""Tool integration test."""

import asyncio
from tool import TOOL_REGISTRY

print(f"Registered tools: {list(TOOL_REGISTRY.keys())}\n")

# Option 1: Manual preload (if not using agent)
# from tool.model_config import preload_tools
# preload_tools(tool_bank=["estimate_region_depth", "estimate_object_depth"])

# Option 2: Tools will be auto-preloaded when creating instances below

IMG = "test_image.png"
# IMG2 = "test_image2.png"
# IMG3 = "test_image3.png"


async def test():
    # calculator
    # r = await TOOL_REGISTRY["calculator"]().call_async({"expression": "2 + 3 * 4"})
    # print(f"calculator: {r}")
    
    # crop
    # r = await TOOL_REGISTRY["crop"]().call_async({"image": IMG, "bbox": [0.1, 0.2, 0.5, 0.6]})
    # print(f"crop: {r[:60]}...")
    
    # zoom_in
    # r = await TOOL_REGISTRY["zoom_in"]().call_async({"image": IMG, "bbox": [0.2, 0.2, 0.8, 0.8], "zoom_factor": 2.0})
    # print(f"zoom_in: {r[:60]}...")
    
    # visualize_regions
    # r = await TOOL_REGISTRY["visualize_regions"]().call_async({
    #     "image_path": IMG, 
    #     "regions": [{"bbox": [0.1, 0.1, 0.5, 0.5], "label": "A"}, {"bbox": [0.5, 0.5, 0.9, 0.9], "label": "B"}]
    # })
    # print(f"visualize_regions: {r[:60]}...")
    
    # solve_math_equation (skip if no API key)
    # r = await TOOL_REGISTRY["solve_math_equation"]().call_async({"query": "solve x^2 - 4 = 0"})
    # print(f"solve_math_equation: {r}")

    # get_objects
    # r = await TOOL_REGISTRY["get_objects"]().call_async({"image": IMG})
    # print(f"get_objects: {r}")

    # localize_objects
    # r = await TOOL_REGISTRY["localize_objects"]().call_async({"image": IMG, "objects": ["bread", "orange"]})
    # print(f"localize_objects: {r}")

    # detect_faces
    # r = await TOOL_REGISTRY["detect_faces"]().call_async({"image": IMG})
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
