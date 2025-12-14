"""Simple test script for MultimodalAgent."""

import asyncio
import sys
import os
import logging

# 配置 logging 显示到终端
logging.basicConfig(level=logging.INFO, format='%(message)s')

sys.path.append('.')

from agent.mm_agent import MultimodalAgent


# Test data paths (use absolute paths)
TEST_IMAGE = os.path.abspath("test_data/000000000009.jpg")
TEST_VIDEO = os.path.abspath("test_data/0A8CF.mp4")


async def test_calculator():
    """Test calculator tool."""
    try:
        agent = MultimodalAgent(
            tool_bank=["calculator"],
            model_name="qwen2.5-vl-72b-instruct",
            max_tokens=2048,
        )
        
        result = await agent.act(query="Calculate 123 * 456 + 789", verbose=True)
        print(f"\n>>> Result: {result}\n")
    except Exception as e:
        print(f"\n>>> Error: {str(e)}\n")
        import traceback
        traceback.print_exc()


async def test_image_zoom():
    """Test image with zoom_in tool."""
    try:
        agent = MultimodalAgent(
            tool_bank=["zoom_in"],
            model_name="qwen2.5-vl-72b-instruct",
            max_tokens=2048,
        )
        
        result = await agent.act(
            query=f"Zoom in on the center region (bbox [0.25, 0.25, 0.75, 0.75]) by 2x of this image: {TEST_IMAGE}",
            images=[TEST_IMAGE],
            verbose=True,
        )
        print(f"\n>>> Result: {result}\n")
    except Exception as e:
        print(f"\n>>> Error: {str(e)}\n")
        import traceback
        traceback.print_exc()


async def test_video():
    """Test video description."""
    try:
        agent = MultimodalAgent(
            tool_bank=None,
            model_name="qwen2.5-vl-72b-instruct",
            max_tokens=2048,
        )
        
        result = await agent.act(
            query="Describe what happens in this video",
            videos=[{"video": TEST_VIDEO, "min_pixels": 4*32*32, "max_pixels": 256*32*32}],
            verbose=True,
        )
        print(f"\n>>> Result: {result}\n")
    except Exception as e:
        print(f"\n>>> Error: {str(e)}\n")
        import traceback
        traceback.print_exc()


async def main():
    """Run tests."""
    tests = [
        ("Calculator", test_calculator),
        ("Image + Zoom", test_image_zoom),
        ("Video", test_video),
    ]
    
    print("\nMultimodalAgent Tests:")
    for i, (name, _) in enumerate(tests, 1):
        print(f"  {i}. {name}")
    
    choice = input("\nSelect (1-3): ").strip()
    
    if choice in ["1", "2", "3"]:
        name, test_func = tests[int(choice) - 1]
        print(f"\n{'='*60}\n{name} Test\n{'='*60}")
        await test_func()


if __name__ == "__main__":
    asyncio.run(main())

