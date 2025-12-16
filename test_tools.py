"""Tool integration test with Memory system."""

import asyncio
import json
from pathlib import Path
from tool import TOOL_REGISTRY
from mm_memory.memory import Memory

print(f"Registered tools: {list(TOOL_REGISTRY.keys())}\n")

# Optional: preload models to avoid first-call latency
from tool.model_cache import preload_tools
preload_tools(tool_bank=["zoom_in", "localize_objects", "estimate_region_depth", "estimate_object_depth"])
print("✅ Preload success\n")

IMG = "test_image.png"

# Output directory
OUTPUT_DIR = Path("test_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Initialize Memory system
memory = Memory(base_dir=str(OUTPUT_DIR))
memory.start_task("test_tools")


async def test():
    """Test tools with Memory system for ID management and observations."""
    
    # Add input image to memory
    img_id = memory.add_input(IMG, modality="img")
    print(f"📸 Input: {IMG} -> {img_id}\n")
    
    # Test 1: zoom_in
    print("🔬 zoom_in")
    tool_params = {"image": img_id, "bbox": [0.2, 0.2, 0.8, 0.8], "zoom_factor": 2.0}
    result = await TOOL_REGISTRY["zoom_in"]().call_async(tool_params)
    
    if isinstance(result, dict) and "output_image" in result:
        output_id = memory.log_action(
            tool="zoom_in",
            properties=tool_params,
            observation=result,
            output_object=result["output_image"],
            output_type="img",
            description="Zoomed in on the specified region"
        )
        print(f"  ✅ {output_id}")
    else:
        print(f"  ❌ {result}")
    
    # Test 2: localize_objects
    print("\n🎯 localize_objects")
    tool_params = {"image": img_id, "objects": ["bread", "orange"]}
    result = await TOOL_REGISTRY["localize_objects"]().call_async(tool_params)
    
    if isinstance(result, dict) and "output_image" in result:
        output_id = memory.log_action(
            tool="localize_objects",
            properties=tool_params,
            observation=result,
            output_object=result["output_image"],
            output_type="img",
            description=f"Localized {tool_params['objects']} in {img_id}"
        )
        print(f"  ✅ {output_id} (found {len(result.get('regions', []))} regions)")
    else:
        print(f"  ❌ {result}")
    
    # Test 3: estimate_region_depth
    print("\n📏 estimate_region_depth")
    tool_params = {"image": img_id, "bbox": [0.1, 0.2, 0.5, 0.6], "mode": "mean"}
    result = await TOOL_REGISTRY["estimate_region_depth"]().call_async(tool_params)
    
    if isinstance(result, dict) and "estimated depth" in result:
        memory.log_action(
            tool="estimate_region_depth",
            properties=tool_params,
            observation=result
        )
        print(f"  ✅ depth={result['estimated depth']}")
    else:
        print(f"  ❌ {result}")
    
    # Test 4: estimate_object_depth
    print("\n📏 estimate_object_depth")
    tool_params = {"image": img_id, "object": "nut"}
    result = await TOOL_REGISTRY["estimate_object_depth"]().call_async(tool_params)
    
    if isinstance(result, dict) and "estimated depth" in result:
        memory.log_action(
            tool="estimate_object_depth",
            properties=tool_params,
            observation=result
        )
        print(f"  ✅ depth={result['estimated depth']}")
    else:
        print(f"  ❌ {result}")
    
    # Save memory trace
    print(f"\n💾 Saved to: {OUTPUT_DIR / 'memory_trace.json'}")
    with open(OUTPUT_DIR / "memory_trace.json", "w") as f:
        json.dump(memory.trace_data, f, indent=2, default=str)

if __name__ == "__main__":
    asyncio.run(test())
