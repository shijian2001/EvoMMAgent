import os
import asyncio
import httpx
import logging
import sys
import requests
import json

sys.path.append('.')

from openai import AsyncOpenAI
from tqdm import tqdm
from typing import Callable, Any, List, Dict, Union, Tuple
from api.utils.key_operator import ApiKeyCycler
from api.utils.multimodal_processor import build_multimodal_message
from dotenv import load_dotenv

load_dotenv('./api/utils/keys.env')
DIRECTLLM_API_KEY_PROGRAM = os.environ.get("DIRECTLLM_API_KEY_PROGRAM", "{}")
DIRECTLLM_API_KEY_PROGRAM = json.loads(DIRECTLLM_API_KEY_PROGRAM)
DIRECTLLM_API_KEY_USER = os.environ.get("DIRECTLLM_API_KEY_USER", "{}")
DIRECTLLM_API_KEY_USER = json.loads(DIRECTLLM_API_KEY_USER)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


class CustomizeChatService:
    """Customizable chat service for various LLM providers.
    
    Supports GPT, Gemini, Qwen, DeepSeek, and other models with retry logic,
    configurable parameters, and flexible API endpoints. Includes support for
    multimodal inputs (text, image, video).
    """
    
    def __init__(
            self,
            model_name: str,
            max_retries: int = 5,
            retry_delay: int = 3,
            max_tokens: int = -1,
            temperature: float = 0.1,
            top_p: float = 0.5,
            timeout: int = 1024,
            use_customize_url: bool = False,
            customize_url: str = "",
            use_api_key: bool = False,
    ):
        """Initialize the chat service with model and request configurations.
        
        Args:
            model_name: Name of the model to use
            max_retries: Maximum number of retry attempts on failure
            retry_delay: Delay in seconds between retries
            max_tokens: Maximum tokens in response (-1 for unlimited)
            temperature: Sampling temperature for generation
            top_p: Nucleus sampling parameter
            timeout: Request timeout in seconds
            use_customize_url: Whether to use a custom API endpoint
            customize_url: Custom API endpoint URL
            use_api_key: Whether to use API key from environment variables
        """
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout

        self.use_customize_url = use_customize_url
        self.customize_url = customize_url
        self.use_api_key = use_api_key

    async def chat_gpt(
            self,
            system_prompt: List[Dict[str, str]],
            user_prompt: List[Dict[str, str]],
            check_func: Callable[[str], Any],
            gpt_api_key: str,
    ) -> str:
        """Call GPT API with retry logic.
        
        Args:
            system_prompt: System message content
            user_prompt: User message content
            check_func: Function to validate and process response
            gpt_api_key: GPT API key for authentication
            
        Returns:
            Model response string, empty string on failure
        """
        for attempt in range(self.max_retries):
            try:
                headers = {
                    "api-key": f"{gpt_api_key}",
                    "Content-Type": "application/json"
                }

                if self.max_tokens > 0:
                    request = {
                        "max_tokens": self.max_tokens,
                        "messages": [
                            {
                                "role": "system",
                                "content": system_prompt
                            },
                            {
                                "role": "user",
                                "content": user_prompt
                            }
                        ],
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                    }
                else:
                    request = {
                        "messages": [
                            {
                                "role": "system",
                                "content": system_prompt
                            },
                            {
                                "role": "user",
                                "content": user_prompt
                            }
                        ],
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                    }

                async with httpx.AsyncClient() as client:
                    if self.timeout > 0:
                        response = await client.post(
                            "https://runway.devops.xiaohongshu.com/openai/chat/completions?api-version=2024-12-01-preview",
                            headers=headers,
                            json=request,
                            timeout=self.timeout,
                        )
                    else:
                        response = await client.post(
                            "https://runway.devops.xiaohongshu.com/openai/chat/completions?api-version=2024-12-01-preview",
                            headers=headers,
                            json=request,
                        )

                if response.status_code != 200:
                    logging.info(f"Failed to call API {response.status_code} - {response.text}")
                    response.raise_for_status()
                response = response.json()["choices"][0]["message"]["content"]
                response = check_func(response)
                return response

            except (
                    httpx.RequestError,
                    Exception,
            ) as e:
                tqdm.write(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                    continue
                else:
                    tqdm.write(f"Reached max retries {self.max_retries}, giving up")
                    return ""

    async def chat_gpt_oss(
            self,
            system_prompt: List[Dict[str, str]],
            user_prompt: List[Dict[str, str]],
            check_func: Callable[[str], Any],
            cycler: ApiKeyCycler,
    ) -> str:
        """Call GPT OSS API with retry logic and key cycling.
        
        Args:
            system_prompt: System message content
            user_prompt: User message content
            check_func: Function to validate and process response
            cycler: API key cycler for load balancing
            
        Returns:
            Model response string, empty string on failure
        """
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        for attempt in range(self.max_retries):
            try:
                api_key = await cycler.get_key()
                client = AsyncOpenAI(
                    api_key=api_key,
                    base_url="http://redservingapi.devops.xiaohongshu.com/v1"
                )
                if self.max_tokens > 0:
                    completion = await client.chat.completions.create(
                        model="gpt-oss-120b",
                        messages=messages,
                        stream=False,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                    )
                else:
                    completion = await client.chat.completions.create(
                        model="gpt-oss-120b",
                        messages=messages,
                        stream=False,
                        temperature=self.temperature,
                    )
                response = completion.choices[0].message.content
                response = check_func(response)
                return response

            except (
                    requests.exceptions.RequestException,
                    Exception,
            ) as e:
                tqdm.write(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay)
                continue
            else:
                tqdm.write(f"Reached max retries {self.max_retries}, giving up")
                return ""

    async def chat_gemini(
            self,
            system_prompt: List[Dict[str, str]],
            user_prompt: List[Dict[str, str]],
            check_func: Callable[[str], Any],
            gemini_api_key: str,
            directllm_api_key: str,
            return_cot: bool = False,
    ) -> str:
        """Call Gemini API with retry logic.
        
        Args:
            system_prompt: System message content
            user_prompt: User message content
            check_func: Function to validate and process response
            gemini_api_key: Gemini API key
            directllm_api_key: DirectLLM API key
            return_cot: Whether to include chain-of-thought reasoning
            
        Returns:
            Model response string, empty string on failure
        """
        for attempt in range(self.max_retries):
            try:
                headers = {'content-type': 'application/json', 'api-key': f'{directllm_api_key}@{gemini_api_key}'}
                if self.max_tokens > 0:
                    request = {
                        "contents": [
                            {"role": "user", "parts": system_prompt},
                            {"role": "user", "parts": user_prompt},
                        ],
                        "generationConfig": {
                            "maxOutputTokens": self.max_tokens,
                            "thinkingConfig": {
                                "includeThoughts": return_cot
                            }
                        }
                    }
                else:
                    request = {
                        "contents": [
                            {"role": "user", "parts": system_prompt},
                            {"role": "user", "parts": user_prompt},
                        ],
                        "generationConfig": {
                            "thinkingConfig": {
                                "includeThoughts": return_cot
                            }
                        }
                    }

                async with httpx.AsyncClient() as client:
                    if self.timeout > 0:
                        response = await client.post(
                            "http://redservingapi.devops.xiaohongshu.com/google/gemini-2.5-pro-preview:generateContent",
                            headers=headers,
                            json=request,
                            timeout=self.timeout,
                        )
                    else:
                        response = await client.post(
                            "http://redservingapi.devops.xiaohongshu.com/google/gemini-2.5-pro-preview:generateContent",
                            headers=headers,
                            json=request,
                        )

                if response.status_code != 200:
                    logging.info(f"Failed to call API {response.status_code} - {response.text}")
                    response.raise_for_status()
                response = response.json()["candidates"][0]["content"]['parts'][0]['text']
                response = check_func(response)
                return response

            except (
                    httpx.RequestError,
                    Exception,
            ) as e:
                tqdm.write(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay)
                continue
            else:
                tqdm.write(f"Reached max retries {self.max_retries}, giving up")
                return ""

    async def chat_qwen_or_deepseek(
            self,
            system_prompt: List[Dict[str, str]],
            user_prompt: List[Dict[str, str]],
            check_func: Callable[[str], Any],
            cycler: ApiKeyCycler,
            return_cot: bool = False
    ) -> Union[str, Tuple[str, str]]:
        """Call Qwen or DeepSeek API with retry logic and key cycling.
        
        Args:
            system_prompt: System message content
            user_prompt: User message content
            check_func: Function to validate and process response
            cycler: API key cycler for load balancing
            return_cot: Whether to return chain-of-thought reasoning
            
        Returns:
            Model response string, or tuple of (response, reasoning) if return_cot is True.
            Empty string on failure.
        """
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        basic_url = self.customize_url if self.use_customize_url and self.customize_url else "http://redservingapi.devops.xiaohongshu.com/v1"
        for attempt in range(self.max_retries):
            try:
                if self.use_api_key and self.model_name in DIRECTLLM_API_KEY_PROGRAM:
                    api_key = DIRECTLLM_API_KEY_PROGRAM[self.model_name]
                else:
                    api_key = await cycler.get_key()

                client = AsyncOpenAI(
                    api_key=api_key,
                    base_url=basic_url
                )
                if self.max_tokens > 0:
                    completion = await client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        stream=False,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                    )
                else:
                    completion = await client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        stream=False,
                        temperature=self.temperature,
                    )

                response = completion.choices[0].message.content
                response = check_func(response)
                if return_cot and "reasoning_content" in completion.choices[0].message:
                    cot = completion.choices[0].message.reasoning_content
                    return (response, cot)
                else:
                    return response

            except (
                    requests.exceptions.RequestException,
                    Exception,
            ) as e:
                tqdm.write(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay)
                continue
            else:
                tqdm.write(f"Reached max retries {self.max_retries}, giving up")
                return ""

    async def chat_qwen_vl_or_deepseek_vl(
            self,
            system_prompt: str,
            user_prompt: List[Dict[str, Any]],
            check_func: Callable[[str], Any],
            cycler: ApiKeyCycler,
            return_cot: bool = False,
            image_patch_size: int = None,
            return_video_metadata: bool = None,
            auto_detect: bool = True
    ) -> Union[str, Tuple[str, str]]:
        """Call Qwen-VL or DeepSeek-VL API with multimodal support.
        
        Supports text, image, and video inputs with automatic format conversion
        and retry logic with key cycling. Can auto-detect Qwen version from model name.
        
        Args:
            system_prompt: System message as plain string
            user_prompt: User message content with multimodal elements (text/image/video)
            check_func: Function to validate and process response
            cycler: API key cycler for load balancing
            return_cot: Whether to return chain-of-thought reasoning
            image_patch_size: Patch size (14 for Qwen2.5-VL, 16 for Qwen3-VL). 
                            If None and auto_detect=True, will detect from model_name
            return_video_metadata: Return video metadata (Qwen3-VL only). 
                                  If None and auto_detect=True, will detect from model_name
            auto_detect: Whether to auto-detect parameters from model name. Default: True
            
        Returns:
            Model response string, or tuple of (response, reasoning) if return_cot is True.
            Empty string on failure.
        """
        # Auto-detect from model name if parameters not explicitly set
        if auto_detect and (image_patch_size is None or return_video_metadata is None):
            processed_content, mm_kwargs = build_multimodal_message(
                user_prompt,
                model_name=self.model_name
            )
        else:
            # Use explicit parameters or defaults
            patch_size = image_patch_size if image_patch_size is not None else 14
            video_metadata = return_video_metadata if return_video_metadata is not None else False
            processed_content, mm_kwargs = build_multimodal_message(
                user_prompt,
                image_patch_size=patch_size,
                return_video_metadata=video_metadata
            )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": processed_content}
        ]

        basic_url = self.customize_url if self.use_customize_url and self.customize_url else "http://redservingapi.devops.xiaohongshu.com/v1"
        for attempt in range(self.max_retries):
            try:
                if self.use_api_key and self.model_name in DIRECTLLM_API_KEY_PROGRAM:
                    api_key = DIRECTLLM_API_KEY_PROGRAM[self.model_name]
                else:
                    api_key = await cycler.get_key()

                client = AsyncOpenAI(
                    api_key=api_key,
                    base_url=basic_url
                )
                
                # Prepare request parameters
                request_params = {
                    "model": self.model_name,
                    "messages": messages,
                    "stream": False,
                    "temperature": self.temperature,
                }
                
                if self.max_tokens > 0:
                    request_params["max_tokens"] = self.max_tokens
                
                # Add multimodal processor kwargs if present
                if mm_kwargs:
                    request_params["extra_body"] = {"mm_processor_kwargs": mm_kwargs}

                completion = await client.chat.completions.create(**request_params)

                response = completion.choices[0].message.content
                print("response: {}".format(response))
                response = check_func(response)
                if return_cot and hasattr(completion.choices[0].message, "reasoning_content"):
                    cot = completion.choices[0].message.reasoning_content
                    return (response, cot)
                else:
                    return response

            except (
                    requests.exceptions.RequestException,
                    Exception,
            ) as e:
                tqdm.write(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay)
                continue
            else:
                tqdm.write(f"Reached max retries {self.max_retries}, giving up")
                return ""

    async def chat_gemini_vl(
            self,
            system_prompt: str,
            user_prompt: List[Dict[str, Any]],
            check_func: Callable[[str], Any],
            cycler: ApiKeyCycler,
            return_cot: bool = False
    ) -> Union[str, Tuple[str, str]]:
        """Call Gemini-VL API with multimodal support.
        
        TODO: Implement Gemini Vision-Language API support
        
        Args:
            system_prompt: System message as plain string
            user_prompt: User message content with multimodal elements (text/image/video)
            check_func: Function to validate and process response
            cycler: API key cycler for load balancing
            return_cot: Whether to return chain-of-thought reasoning
            
        Returns:
            Model response string, or tuple of (response, reasoning) if return_cot is True.
            Empty string on failure.
        """
        # TODO: Implement Gemini VL multimodal processing
        # This should handle image and video inputs similar to chat_qwen_vl_or_deepseek_vl
        # but adapted for Gemini's API format
        raise NotImplementedError("Gemini VL support is not yet implemented. TODO: Add implementation.")
    
    async def chat_gpt_vl(
            self,
            system_prompt: str,
            user_prompt: List[Dict[str, Any]],
            check_func: Callable[[str], Any],
            cycler: ApiKeyCycler,
            return_cot: bool = False
    ) -> Union[str, Tuple[str, str]]:
        """Call GPT-VL (GPT-4V/GPT-4o) API with multimodal support.
        
        TODO: Implement GPT Vision-Language API support
        
        Args:
            system_prompt: System message as plain string
            user_prompt: User message content with multimodal elements (text/image/video)
            check_func: Function to validate and process response
            cycler: API key cycler for load balancing
            return_cot: Whether to return chain-of-thought reasoning
            
        Returns:
            Model response string, or tuple of (response, reasoning) if return_cot is True.
            Empty string on failure.
        """
        # TODO: Implement GPT-4V/GPT-4o multimodal processing
        # This should handle image and video inputs
        # GPT-4V API format uses image_url in message content
        raise NotImplementedError("GPT VL support is not yet implemented. TODO: Add implementation.")


if __name__ == "__main__":
    async def main():
        """Demo function showing how to use CustomizeChatService."""
        model_name = "gemini-2.5-pro"
        service = CustomizeChatService(
            model_name=model_name,
            max_retries=5,
            retry_delay=1,
            temperature=0.1,
        )

        def check_func(response: str) -> str:
            return response

        _system_prompt = "You need to act as an intelligent assistant."
        _user_prompt = "Who are you?"
        if "qwen" in model_name or "deepseek" in model_name or "qwq" in model_name:
            system_prompt = [
                {
                    "type": "text",
                    "text": _system_prompt
                }
            ]
            user_prompt = [
                {
                    "type": "text",
                    "text": _user_prompt
                }
            ]
            cycler = ApiKeyCycler(api_key_list=list(DIRECTLLM_API_KEY_USER.values()))
            response = await service.chat_qwen_or_deepseek(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                check_func=check_func,
                cycler=cycler,
                return_cot=False,
            )

            print(response)

        if "gemini" in model_name:
            system_prompt = [
                {
                    "text": _system_prompt
                }
            ]
            user_prompt = [
                {
                    "text": _user_prompt
                }
            ]

            response = await service.chat_gemini(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                check_func=check_func,
                gemini_api_key=GEMINI_API_KEY,
                directllm_api_key=DIRECTLLM_API_KEY_USER["tusen"],
                return_cot=False,
            )
            print(response)
        
        # Example for VL models (Qwen-VL, DeepSeek-VL)
        # Uncomment to test multimodal processing
        """
        if "vl" in model_name.lower():
            cycler = ApiKeyCycler(api_key_list=list(DIRECTLLM_API_KEY_USER.values()))
            
            # Example 1: Image with resized dimensions (auto-detect version from model_name)
            user_prompt_image = [
                {"type": "text", "text": "Describe this image."},
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                    "resized_height": 280,
                    "resized_width": 420,
                }
            ]
            
            # Auto-detect: system will detect Qwen2.5-VL or Qwen3-VL from model_name
            response = await service.chat_qwen_vl_or_deepseek_vl(
                system_prompt="You are a helpful assistant.",
                user_prompt=user_prompt_image,
                check_func=check_func,
                cycler=cycler,
                auto_detect=True,  # Default: automatically detect from model name
            )
            print(response)
            
            # Example 2: Video with explicit parameters (Qwen3-VL)
            # user_prompt_video = [
            #     {"type": "text", "text": "Describe this video."},
            #     {
            #         "type": "video",
            #         "video": "https://example.com/video.mp4",
            #         "min_pixels": 4 * 32 * 32,
            #         "max_pixels": 256 * 32 * 32,
            #         "total_pixels": 20480 * 32 * 32,
            #     }
            # ]
            # 
            # # Manual mode: explicitly set parameters
            # response = await service.chat_qwen_vl_or_deepseek_vl(
            #     system_prompt="You are a helpful assistant.",
            #     user_prompt=user_prompt_video,
            #     check_func=check_func,
            #     cycler=cycler,
            #     auto_detect=False,
            #     image_patch_size=16,  # Qwen3-VL
            #     return_video_metadata=True,  # Qwen3-VL only
            # )
        """


    asyncio.run(main())
