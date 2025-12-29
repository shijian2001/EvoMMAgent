"""Test script for new API package."""

import asyncio
import sys
import os

sys.path.append('.')

from api import APIPool, load_api_keys


async def test_text_only():
    """Test text-only query."""
    print("\n" + "="*60)
    print("Test 1: Text-only Query")
    print("="*60)
    
    try:
        # Load API keys
        api_keys = load_api_keys()
        
        # Create API pool
        api_pool = APIPool(
            model_name="qwen2.5-vl-72b-instruct",
            api_keys=api_keys,
            max_concurrent_per_key=10,
        )
        
        # Test query
        result = await api_pool.execute(
            "qa",
            system="You are a helpful assistant.",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
        )
        
        print(f"\nResult: {result['answer']}")
        print(f"\nStats: {await api_pool.get_stats()}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()


async def test_image():
    """Test image query."""
    print("\n" + "="*60)
    print("Test 2: Image Query")
    print("="*60)
    
    try:
        # Load API keys
        api_keys = load_api_keys()
        
        # Create API pool
        api_pool = APIPool(
            model_name="qwen2.5-vl-72b-instruct",
            api_keys=api_keys,
            max_concurrent_per_key=10,
        )

        print("Successfully created API pool")
        
        # Test image path
        test_image = os.path.abspath("test_data/000000000009.jpg")
        
        print("Starting test query with image")
        
        result = await api_pool.execute(
            "qa",
            system="You are a helpful assistant that can understand images.",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in detail."},
                        {"type": "image", "image": test_image}
                    ]
                }
            ],
        )
        
        print(f"\nResult: {result['answer']}")
        print(f"\nStats: {await api_pool.get_stats()}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()


async def test_video():
    """Test video query."""
    print("\n" + "="*60)
    print("Test 3: Video Query")
    print("="*60)
    
    try:
        # Load API keys
        api_keys = load_api_keys()
        
        # Create API pool
        api_pool = APIPool(
            model_name="qwen2.5-vl-72b-instruct",
            api_keys=api_keys,
            max_concurrent_per_key=10,
        )
        
        # Test video path
        test_video = os.path.abspath("test_data/0A8CF.mp4")
        
        result = await api_pool.execute(
            "qa",
            system="You are a helpful assistant that can understand videos.",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe what happens in this video."},
                        {
                            "type": "video",
                            "video": test_video,
                            "min_pixels": 4 * 32 * 32,
                            "max_pixels": 256 * 32 * 32
                        }
                    ]
                }
            ],
        )
        
        print(f"\nResult: {result['answer']}")
        print(f"\nStats: {await api_pool.get_stats()}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all tests."""
    tests = [
        ("Text-only Query", test_text_only),
        ("Image Query", test_image),
        ("Video Query", test_video),
    ]
    
    print("\nAPI Test Suite:")
    for i, (name, _) in enumerate(tests, 1):
        print(f"  {i}. {name}")
    
    choice = input("\nSelect test (1-3, or 'all'): ").strip()
    
    if choice.lower() == "all":
        for name, test_func in tests:
            await test_func()
    elif choice in ["1", "2", "3"]:
        name, test_func = tests[int(choice) - 1]
        await test_func()
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    asyncio.run(main())
