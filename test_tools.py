"""Tool integration test with Memory system.

This test framework provides a convenient way to test tools with automatic
ID management and observation formatting.

To add a new tool test:
1. Add tool name to TOOLS_TO_PRELOAD if it uses models
2. Call test_tool() in the test() function with appropriate parameters
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, Any, Optional
from tool import TOOL_REGISTRY
from mm_memory.memory import Memory

print(f"Registered tools: {list(TOOL_REGISTRY.keys())}\n")

# Configuration
IMG = "test_image.png"
OUTPUT_DIR = Path("test_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Tools that need model preloading
TOOLS_TO_PRELOAD = ["zoom_in", "localize_objects", "estimate_region_depth", "estimate_object_depth"]

# Optional: preload models to avoid first-call latency
from tool.model_cache import preload_tools
preload_tools(tool_bank=TOOLS_TO_PRELOAD)
print("✅ Preload success\n")

# Initialize Memory system
memory = Memory(base_dir=str(OUTPUT_DIR))
memory.start_task("test_tools")


async def test_tool(
    tool_name: str,
    params: Dict[str, Any],
    emoji: str = "🔧",
    extract_info: Optional[callable] = None
) -> Optional[str]:
    """Test a tool with automatic ID resolution and Memory integration.
    
    Args:
        tool_name: Name of the tool to test
        params: Tool parameters (with IDs, not paths)
        emoji: Emoji for display
        extract_info: Optional function to extract additional info from result
                     Signature: (result: Dict) -> str
    
    Returns:
        Output ID if successful, None otherwise
        
    Example:
        await test_tool(
            "zoom_in",
            {"image": img_id, "bbox": [0.2, 0.2, 0.8, 0.8], "zoom_factor": 2.0},
            emoji="🔬"
        )
    """
    print(f"{emoji} {tool_name}")
    
    # Step 1: Resolve IDs to paths (Input: img_0 -> path)
    params_resolved = memory.resolve_ids(params)
    
    # Step 2: Call tool with resolved paths
    try:
        result = await TOOL_REGISTRY[tool_name]().call_async(params_resolved)
    except Exception as e:
        print(f"  ❌ Exception: {e}")
        return None
    
    # Step 3: Handle result
    if not isinstance(result, dict):
        print(f"  ❌ Unexpected result type: {type(result)}")
        return None
    
    if "error" in result:
        print(f"  ❌ {result['error']}")
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
        
        # Print result with optional info
        info_str = f" ({extract_info(result)})" if extract_info and callable(extract_info) else ""
        print(f"  ✅ {output_id}{info_str}")
        return output_id
    else:
        memory.log_action(
            tool=tool_name,
            properties=params,
            observation=observation_data
        )
        
        # Print result with optional info
        info_str = f" ({extract_info(result)})" if extract_info and callable(extract_info) else ""
        print(f"  ✅{info_str}")
        return None


async def test():
    """Test tools with Memory system for ID management and observations.
    
    To add a new tool test, simply call:
        await test_tool("tool_name", params, emoji="🔧")
    """
    
    # Add input image to memory
    img_id = memory.add_input(IMG, modality="img")
    print(f"📸 Input: {IMG} -> {img_id}\n")
    
    # Test 1: zoom_in
    await test_tool(
        "zoom_in",
        {"image": img_id, "bbox": [0.2, 0.2, 0.8, 0.8], "zoom_factor": 2.0},
        emoji="🔬"
    )
    
    # Test 2: localize_objects
    await test_tool(
        "localize_objects",
        {"image": img_id, "objects": ["bread", "orange"]},
        emoji="🎯",
        extract_info=lambda r: f"found {len(r.get('regions', []))} regions"
    )
    
    # Test 3: estimate_region_depth
    await test_tool(
        "estimate_region_depth",
        {"image": img_id, "bbox": [0.1, 0.2, 0.5, 0.6], "mode": "mean"},
        emoji="📏",
        extract_info=lambda r: f"depth={r.get('estimated depth')}"
    )
    
    # Test 4: estimate_object_depth
    await test_tool(
        "estimate_object_depth",
        {"image": img_id, "object": "nut"},
        emoji="📏",
        extract_info=lambda r: f"depth={r.get('estimated depth')}"
    )
    
    # ========================================
    # Add your own tool tests here!
    # Example:
    # await test_tool(
    #     "your_tool_name",
    #     {"image": img_id, "param1": value1},
    #     emoji="✨"
    # )
    # ========================================
    
    # Save memory trace
    print(f"\n💾 Saved to: {OUTPUT_DIR / 'memory_trace.json'}")
    with open(OUTPUT_DIR / "memory_trace.json", "w") as f:
        json.dump(memory.trace_data, f, indent=2, default=str)
    
    # Print summary
    print(f"\n📊 Summary:")
    print(f"  Actions: {len(memory.trace_data['actions'])}")
    print(f"  Images: {len(memory.trace_data.get('images', []))}")

if __name__ == "__main__":
    """
    Usage:
        python test_tools.py
        
    Adding New Tool Tests:
        1. (Optional) Add tool to TOOLS_TO_PRELOAD if it uses models
        2. In test() function, call:
           
           await test_tool(
               "your_tool_name",
               {"image": img_id, "param": value},
               emoji="✨",
               extract_info=lambda r: f"info={r.get('key')}"  # optional
           )
        
    The test_tool() function automatically handles:
        - ID to path resolution (input)
        - Path to ID resolution (output/observation)
        - Memory logging
        - Error handling
        - Result formatting
    """
    asyncio.run(test())
