"""Tool integration test."""

import asyncio
from tool import TOOL_REGISTRY

print(f"Registered: {list(TOOL_REGISTRY.keys())}\n")

IMG = "test_data/000000000009.jpg"


async def test():
    # calculator
    r = await TOOL_REGISTRY["calculator"]().call_async({"expression": "2 + 3 * 4"})
    print(f"calculator: {r}")
    
    # crop
    r = await TOOL_REGISTRY["crop"]().call_async({"image": IMG, "bbox": [0.1, 0.2, 0.5, 0.6]})
    print(f"crop: {r[:60]}...")
    
    # zoom_in
    r = await TOOL_REGISTRY["zoom_in"]().call_async({"image": IMG, "bbox": [0.2, 0.2, 0.8, 0.8], "zoom_factor": 2.0})
    print(f"zoom_in: {r[:60]}...")
    
    # visualize_regions
    r = await TOOL_REGISTRY["visualize_regions"]().call_async({
        "image_path": IMG, 
        "regions": [{"bbox": [0.1, 0.1, 0.5, 0.5], "label": "A"}, {"bbox": [0.5, 0.5, 0.9, 0.9], "label": "B"}]
    })
    print(f"visualize_regions: {r[:60]}...")
    
    # solve_math_equation (skip if no API key)
    # r = await TOOL_REGISTRY["solve_math_equation"]().call_async({"query": "solve x^2 - 4 = 0"})
    # print(f"solve_math_equation: {r}")


if __name__ == "__main__":
    asyncio.run(test())
