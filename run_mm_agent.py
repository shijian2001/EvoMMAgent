"""Test script for MultimodalAgent with Memory system."""

import asyncio
import sys
import os
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(message)s')

sys.path.append('.')

from agent.mm_agent import MultimodalAgent


# Test data paths
# Note: images can be file paths (str), PIL.Image objects, or dicts with image params
# Example: Load from BLINK dataset - TEST_IMAGE = load_dataset("./data/eval/image/BLINK")["test"][0]["images"][0]
TEST_IMAGE = os.path.abspath("test_image.png")


async def test_calculator_with_memory():
    """Test calculator tool with memory."""
    print("\n" + "="*80)
    print("Test 1: Calculator with Memory")
    print("="*80)
    
    try:
        agent = MultimodalAgent(
            tool_bank=["calculator"],
            model_name="qwen2.5-vl-72b-instruct",
            max_tokens=2048,
            enable_memory=True,
            memory_dir="memory"
        )
        
        result = await agent.act(
            query="Calculate 123 * 456 + 789", 
            verbose=True,
            return_history=True
        )
        
        print(f"\n{'='*80}")
        print("Result:")
        print(f"{'='*80}")
        print(f"Task ID: {result.get('task_id')}")
        print(f"Success: {result.get('success')}")
        print(f"Response: {result.get('response')}")
        
        # Check trace file
        task_id = result.get('task_id')
        if task_id:
            trace_path = f"memory/tasks/{task_id}/trace.json"
            if os.path.exists(trace_path):
                print(f"\n{'='*80}")
                print("Trace file:")
                print(f"{'='*80}")
                with open(trace_path, 'r') as f:
                    trace = json.load(f)
                print(json.dumps(trace, indent=2))
            else:
                print(f"\n⚠️ Trace file not found: {trace_path}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


async def test_crop_with_memory():
    """Test crop tool with memory and image."""
    print("\n" + "="*80)
    print("Test 2: Crop with Memory")
    print("="*80)
    
    if not os.path.exists(TEST_IMAGE):
        print(f"⚠️ Test image not found: {TEST_IMAGE}")
        print("Creating a test image...")
        from PIL import Image
        img = Image.new('RGB', (400, 400), color='red')
        img.save(TEST_IMAGE)
        print(f"✓ Created test image: {TEST_IMAGE}")
    
    try:
        agent = MultimodalAgent(
            tool_bank=["crop"],
            model_name="qwen2.5-vl-72b-instruct",
            max_tokens=2048,
            enable_memory=True,
            memory_dir="memory"
        )
        
        result = await agent.act(
            query="Crop the center region (bbox [0.25, 0.25, 0.75, 0.75]) from this image",
            images=[TEST_IMAGE],
            verbose=True,
            return_history=True
        )
        
        print(f"\n{'='*80}")
        print("Result:")
        print(f"{'='*80}")
        print(f"Task ID: {result.get('task_id')}")
        print(f"Success: {result.get('success')}")
        print(f"Response: {result.get('response')}")
        
        # Check trace and files
        task_id = result.get('task_id')
        if task_id:
            task_dir = f"memory/tasks/{task_id}"
            trace_path = f"{task_dir}/trace.json"
            
            if os.path.exists(trace_path):
                print(f"\n{'='*80}")
                print("Trace file:")
                print(f"{'='*80}")
                with open(trace_path, 'r') as f:
                    trace = json.load(f)
                print(json.dumps(trace, indent=2))
                
                print(f"\n{'='*80}")
                print("Task directory files:")
                print(f"{'='*80}")
                if os.path.exists(task_dir):
                    for f in os.listdir(task_dir):
                        fpath = os.path.join(task_dir, f)
                        size = os.path.getsize(fpath)
                        print(f"  - {f} ({size} bytes)")
            else:
                print(f"\n⚠️ Trace file not found: {trace_path}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


async def test_both():
    """Run both tests sequentially."""
    await test_calculator_with_memory()
    print("\n" + "="*80 + "\n")
    await test_crop_with_memory()


async def main():
    """Run tests."""
    tests = [
        ("Calculator with Memory", test_calculator_with_memory),
        ("Crop with Memory", test_crop_with_memory),
        ("Both Tests", test_both),
    ]
    
    print("\n" + "="*80)
    print("MultimodalAgent Memory System Tests")
    print("="*80)
    print("\nAvailable tests:")
    for i, (name, _) in enumerate(tests, 1):
        print(f"  {i}. {name}")
    
    choice = input("\nSelect test (1-3): ").strip()
    
    if choice in ["1", "2", "3"]:
        name, test_func = tests[int(choice) - 1]
        print(f"\n{'='*80}")
        print(f"Running: {name}")
        print(f"{'='*80}")
        await test_func()
        
        print(f"\n{'='*80}")
        print("Test completed!")
        print(f"{'='*80}")
        print("\nCheck the memory directory for saved traces and files:")
        print("  memory/tasks/000001/")
        print("  memory/tasks/000002/")
    else:
        print("Invalid choice")


if __name__ == "__main__":
    asyncio.run(main())

