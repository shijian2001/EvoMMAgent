"""Simple test script for VL (Vision-Language) API with image and video inputs."""

import asyncio
import sys

sys.path.append('.')

from api.custom_service import CustomizeChatService, DIRECTLLM_API_KEY_USER
from api.utils.key_operator import ApiKeyCycler


async def test_image():
    """Test image processing with Qwen2.5-VL."""
    print("=" * 60)
    print("Testing Image Processing")
    print("=" * 60)
    
    model_name = "qwen2.5-vl-72b-instruct"
    service = CustomizeChatService(
        model_name=model_name,
        max_retries=3,
        retry_delay=1,
        temperature=0.1,
    )
    
    def check_func(response: str) -> str:
        return response
    
    # TODO: Replace with your actual image path
    image_path = "test_data/000000000009.jpg"
    
    user_prompt = [
        {"type": "text", "text": "Describe this image in detail."},
        {
            "type": "image",
            "image": image_path,
            "min_pixels": 50176,
            "max_pixels": 50176,
        }
    ]
    
    cycler = ApiKeyCycler(api_key_list=list(DIRECTLLM_API_KEY_USER.values()))
    
    try:
        response = await service.chat_qwen_vl_or_deepseek_vl(
            system_prompt="You are a helpful visual assistant.",
            user_prompt=user_prompt,
            check_func=check_func,
            cycler=cycler,
            auto_detect=True,
        )
        
        print(f"\n✅ Image Test Response:\n{response}\n")
    except Exception as e:
        print(f"\n❌ Image Test Failed: {str(e)}\n")


async def test_video():
    """Test video processing with Qwen2.5-VL."""
    print("=" * 60)
    print("Testing Video Processing")
    print("=" * 60)
    
    model_name = "qwen2.5-vl-72b-instruct"
    service = CustomizeChatService(
        model_name=model_name,
        max_retries=3,
        retry_delay=1,
        temperature=0.1,
    )
    
    def check_func(response: str) -> str:
        return response
    
    # TODO: Replace with your actual video path
    video_path = "test_data/0A8CF.mp4"
    
    user_prompt = [
        {"type": "text", "text": "Describe what happens in this video."},
        {
            "type": "video",
            "video": video_path,
            "min_pixels": 4 * 32 * 32,
            "max_pixels": 256 * 32 * 32,
        }
    ]
    
    cycler = ApiKeyCycler(api_key_list=list(DIRECTLLM_API_KEY_USER.values()))
    
    try:
        response = await service.chat_qwen_vl_or_deepseek_vl(
            system_prompt="You are a helpful visual assistant.",
            user_prompt=user_prompt,
            check_func=check_func,
            cycler=cycler,
            auto_detect=True,
        )
        
        print(f"\n✅ Video Test Response:\n{response}\n")
    except Exception as e:
        print(f"\n❌ Video Test Failed: {str(e)}\n")


async def main():
    """Run all VL API tests."""
    print("\n" + "=" * 60)
    print("Starting VL API Tests with qwen2.5-vl-72b-instruct")
    print("=" * 60 + "\n")
    
    await test_image()
    await test_video()
    
    print("=" * 60)
    print("All Tests Completed")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())

