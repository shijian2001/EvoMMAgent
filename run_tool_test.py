"""Tool integration test with Memory system.

This test framework provides a convenient way to test tools with automatic
ID management and observation formatting.

To add a new tool test:
1. Add tool name to TOOLS_TO_PRELOAD if it uses models
2. Call test_tool() in the test() function with appropriate parameters
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from tool import TOOL_REGISTRY
from mm_memory.memory import Memory

print(f"Registered tools: {list(TOOL_REGISTRY.keys())}\n")

# Configuration
IMG = "/mnt/tidalfs-bdsz01/usr/wangshijian/EvoMMAgent/test_ocr.jpg"
OUTPUT_DIR = Path("test_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Tools that need model preloading
TOOLS_TO_PRELOAD = [
    "zoom_in", 
    "calculator", 
    "crop", 
    "ocr",
    "visualize_regions",
    "localize_objects", 
    "estimate_region_depth", 
    "estimate_object_depth", 
    "get_image2images_similarity", 
    "get_image2texts_similarity", 
    "get_text2images_similarity",
    "solve_math_equation",
    "web_search"
]

# Optional: preload models to avoid first-call latency
from tool.model_cache import preload_tools
preload_tools(tool_bank=TOOLS_TO_PRELOAD)
print("✅ Preload success\n")

# Initialize Memory system
memory = Memory(base_dir=str(OUTPUT_DIR))
memory.start_task("test_tools")


def _format_observation_data(data: Dict[str, Any]) -> str:
    """Format observation data like mm_agent does for LLM.
    
    Args:
        data: Tool observation data (dict)
        
    Returns:
        Formatted string: "Key1: value1, Key2: value2"
    """
    if not data:
        return "success"
    
    parts = []
    for key, value in data.items():
        # Skip internal fields
        if key in ["id", "type", "description"]:
            continue
            
        # Capitalize first letter of key
        formatted_key = key[0].upper() + key[1:] if key else key
        
        # Format value
        if isinstance(value, list):
            # Special handling for regions (list of dicts with label and bbox_2d)
            if value and isinstance(value[0], dict) and "label" in value[0] and ("bbox_2d" in value[0] or "bbox" in value[0]):
                region_strs = []
                for item in value:
                    label = item["label"]
                    bbox = item.get("bbox_2d", item.get("bbox"))
                    region_strs.append(f"{label} at {bbox}")
                formatted_value = ", ".join(region_strs)
            else:
                formatted_value = str(value)
        else:
            formatted_value = str(value)
        
        parts.append(f"{formatted_key}: {formatted_value}")
    
    return ", ".join(parts)


async def test_tool(
    tool_name: str,
    params: Dict[str, Any]
) -> Optional[str]:
    """Test a tool with automatic ID resolution and Memory integration.
    
    Args:
        tool_name: Name of the tool to test
        params: Tool parameters (with IDs, not paths)
    
    Returns:
        Output ID if successful, None otherwise
        
    Example:
        await test_tool(
            "zoom_in",
            {"image": img_id, "bbox_2d": [0.2, 0.2, 0.8, 0.8], "zoom_factor": 2.0}
        )
    """
    print(f"\n{'='*60}")
    print(f"Testing: {tool_name}")
    print(f"{'='*60}")
    print(f"Input params: {params}")
    
    # Step 1: Resolve IDs to paths (Input: img_0 -> path)
    params_resolved = memory.resolve_ids(params)
    
    # Step 2: Call tool with resolved paths
    try:
        result = await TOOL_REGISTRY[tool_name]().call_async(params_resolved)
    except Exception as e:
        print(f"\n[ERROR] Exception: {e}")
        print("-" * 60)
        return None
    
    # Step 3: Handle result
    if not isinstance(result, dict):
        print(f"\n[ERROR] Unexpected result type: {type(result)}")
        print("-" * 60)
        return None
    
    if "error" in result:
        print(f"\n[ERROR] {result['error']}")
        print("-" * 60)
        return None
    
    # Step 4: Check for multimodal output
    output_object = None
    output_type = None
    
    if "output_image" in result:
        output_object = result["output_image"]
        output_type = "img"
    elif "output_video" in result:
        output_object = result["output_video"]
        output_type = "vid"
    
    # Step 5: Extract observation data (excluding multimodal outputs)
    observation_data = {k: v for k, v in result.items() 
                       if k not in ["output_image", "output_video"]}
    
    # Step 6: Resolve any paths in observation to IDs (Output: path -> img_1)
    observation_data = memory.resolve_paths_to_ids(observation_data)
    
    # Step 7: Get tool description
    from tool.base_tool import TOOL_REGISTRY as TOOL_REG
    tool_instance = TOOL_REG.get(tool_name)
    if tool_instance and output_object:
        description = tool_instance().generate_description(params, observation_data)
    else:
        description = f"{tool_name} output"
    
    # Step 8: Log to memory
    if output_object and output_type:
        output_id = memory.log_action(
            tool=tool_name,
            properties=params,
            observation=observation_data,
            output_object=output_object,
            output_type=output_type,
            description=description
        )
        
        # Format observation like mm_agent does for LLM
        obs_parts = [f"Saved as {output_id}: {description}"]
        if observation_data:
            formatted_data = _format_observation_data(observation_data)
            obs_parts.append(formatted_data)
        observation_str = ". ".join(obs_parts)
        
        # Print formatted observation
        print(f"\n[SUCCESS] Output ID: {output_id}")
        print(f"Observation: {observation_str}")
        print("-" * 60)
        return output_id
    else:
        memory.log_action(
            tool=tool_name,
            properties=params,
            observation=observation_data
        )
        
        # Format observation
        observation_str = _format_observation_data(observation_data) if observation_data else "success"
        
        # Print formatted observation
        print(f"\n[SUCCESS]")
        print(f"Observation: {observation_str}")
        print("-" * 60)
        return None


async def test():
    """Test tools with Memory system for ID management and observations.
    
    To add a new tool test, simply call:
        await test_tool("tool_name", params)
    """
    
    # Add input image to memory
    img_id = memory.add_input(IMG, modality="img")
    print(f"Input image: {IMG} -> {img_id}")
    
    # zoomed_img_id = await test_tool(
    #         "zoom_in",
    #         {"image": img_id, "bbox_2d": [0.2, 0.2, 0.8, 0.8], "zoom_factor": 2.0}
    # )
    
    # cropped_img_id = await test_tool(
    #     "crop",
    #     {"image": img_id, "bbox_2d": [0.1, 0.3, 0.6, 0.7]}
    # )
    
    # await test_tool(
    #     "calculator",
    #     {"expression": "123 * 456 + 789"}
    # )

    await test_tool(
        "solve_math_equation",
        {"query": "x^2 + 2x + 1 = 0, what is x?"}
    )
    
    await test_tool(
        "ocr",
        {"image": img_id}
    )

    await test_tool(
        "web_search",
        {"query": "Which city is the capital of China?", "top_k": 5},
    )
    
    # await test_tool(
    #     "localize_objects",
    #     {"image": img_id, "objects": ["bread", "orange"]}
    # )
    
    # await test_tool(
    #     "estimate_region_depth",
    #     {"image": img_id, "bbox_2d": [0.1, 0.2, 0.5, 0.6], "mode": "mean"}
    # )
    
    # await test_tool(
    #     "estimate_object_depth",
    #     {"image": img_id, "object": "orange"}
    # )
    
    # await test_tool(
    #     "visualize_regions",
    #     {
    #         "image": img_id,
    #         "regions": [
    #             {"bbox_2d": [0.1, 0.1, 0.4, 0.4], "label": "Region A"},
    #             {"bbox_2d": [0.6, 0.6, 0.9, 0.9], "label": "Region B"}
    #         ]
    #     }
    # )
    
    # await test_tool(
    #     "get_image2texts_similarity",
    #     {"image": img_id, "texts": ["a cat", "a dog", "food"]}
    # )
    
    # if zoomed_img_id and cropped_img_id:
    #     await test_tool(
    #         "get_image2images_similarity",
    #         {"image": img_id, "other_images": [zoomed_img_id, cropped_img_id]}
    #     )
    
    # if zoomed_img_id and cropped_img_id:
    #     await test_tool(
    #         "get_text2images_similarity",
    #         {"text": "a zoomed in image", "images": [img_id, zoomed_img_id, cropped_img_id]}
    #     )
    
    # Finalize task and save trace automatically
    memory.end_task(success=True)
    
    # Print summary
    trace_path = Path(memory.task_dir) / "trace.json"
    print(f"\n{'='*60}")
    print(f"Test Summary")
    print(f"{'='*60}")
    print(f"✅ Trace saved to: {trace_path}")
    print(f"Task ID: {memory.task_id}")
    print(f"Total actions: {len(memory.trace_data.get('trace', []))}")
    print(f"Input images: {len(memory.trace_data.get('input', {}).get('images', []))}")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")

if __name__ == "__main__":
    """
    Usage:
        python test_tools.py
        
    Adding New Tool Tests:
        1. (Optional) Add tool to TOOLS_TO_PRELOAD if it uses models
        2. In test() function, call:
           
           await test_tool(
               "your_tool_name",
               {"image": img_id, "param": value}
           )
        
    The test_tool() function automatically handles:
        - ID to path resolution (input)
        - Tool execution
        - Path to ID resolution (output/observation)
        - Memory logging
        - Error handling
        - Complete observation printing
    """
    asyncio.run(test())