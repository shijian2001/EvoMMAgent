from openai import AsyncOpenAI
from typing import Dict, Any, List, Union, Optional
from .vision_utils import build_multimodal_message
from .json_parser import JSONParser
import logging
import asyncio
import time

logger = logging.getLogger(__name__)

# Suppress verbose HTTP request logs from httpx and openai
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)


class QAWrapper:
    """Asynchronous wrapper for LLM API client with automatic retry and error handling."""

    SUPPORTED_REASONING_MODELS = ["DeepSeek-R1", "deepseek-r1"]

    def __init__(
        self, 
        model_name: str, 
        api_key: str, 
        base_url: str = "http://redservingapi.devops.xiaohongshu.com/v1",
        max_retries: int = 5,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        parse_json: bool = True,
    ):
        """
        Initialize an async API wrapper instance.

        Args:
            model_name: Name of the model to use
            api_key: API key for authentication
            base_url: Base URL for the API endpoint
            max_retries: Maximum number of retry attempts for failed requests
            temperature: Sampling temperature for generation
            max_tokens: Maximum tokens in response (None for unlimited)
            parse_json: Whether to automatically parse JSON responses (default: True)
        """
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.parse_json = parse_json

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        self.stats = {
            "calls": 0,
            "errors": 0,
            "retries": 0,
            "total_retry_time": 0.0
        }

    async def qa(
        self, 
        system_prompt: str, 
        user_prompt: Union[str, List[Dict[str, Any]]], 
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        rational: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Send a prompt to the model and get a response.

        Args:
            system_prompt: System message
            user_prompt: User query content. Can be:
                - str: simple text query
                - List[Dict]: multimodal content items [{"type": "text", ...}, {"type": "image", ...}]
                  OR full message history [{"role": "user", ...}, {"role": "assistant", ...}, ...]
            tools: Optional list of tool definitions in OpenAI format
            tool_choice: Tool choice mode ("auto", "required", "none")
            rational: Whether to enable deep reasoning mode
            temperature: Override default temperature
            max_tokens: Override default max_tokens

        Returns:
            Dict with "answer", "rational", and optionally "tool_calls" keys

        Raises:
            ValueError: If reasoning is requested but not supported by the model
            Exception: If all retries are exhausted
        """
        start_time = time.time()
        
        if rational and self.model_name.lower() not in [m.lower() for m in self.SUPPORTED_REASONING_MODELS]:
            raise ValueError(f"Model {self.model_name} does not support reasoning")

        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                
                # Determine if input is multimodal/messages
                if isinstance(user_prompt, list):
                    result = await self._qa_multimodal(
                        system_prompt, 
                        user_prompt,
                        tools=tools,
                        tool_choice=tool_choice,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                else:
                    if rational:
                        result = await self._qa_with_reasoning(
                            system_prompt, 
                            user_prompt,
                            tools=tools,
                            tool_choice=tool_choice,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                    else:
                        result = await self._qa_standard(
                            system_prompt, 
                            user_prompt,
                            tools=tools,
                            tool_choice=tool_choice,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                
                # Log successful request time
                elapsed = time.time() - start_time
                logger.info(f"✓ API call successful in {elapsed:.2f}s (attempt {attempt + 1})")
                return result

            except Exception as e:
                last_exception = e
                self.stats["errors"] += 1
                
                # Log detailed error information
                error_type = type(e).__name__
                error_msg = str(e)
                elapsed = time.time() - start_time
                
                if attempt < self.max_retries - 1:
                    self.stats["retries"] += 1
                    # Gentle exponential backoff: 0.5, 1.0, 2.0, 3.0, 5.0 seconds
                    retry_delay = min(0.5 * (1.5 ** attempt), 5.0)
                    self.stats["total_retry_time"] += retry_delay
                    
                    logger.warning(
                        f"✗ API call failed after {elapsed:.2f}s (attempt {attempt + 1}/{self.max_retries}): "
                        f"{error_type}: {error_msg} - Retrying in {retry_delay:.1f}s"
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(
                        f"✗ API call failed after {elapsed:.2f}s and {self.max_retries} attempts: "
                        f"{error_type}: {error_msg}"
                    )
        
        # All retries exhausted
        raise Exception(f"API call failed after {self.max_retries} retries. Last error: {str(last_exception)}")

    async def _qa_standard(
        self, 
        system_prompt: str, 
        user_prompt: str,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute a standard query without reasoning."""
        request_params = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "temperature": temperature if temperature is not None else self.temperature,
        }
        
        # Add max_tokens if specified
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        if tokens is not None:
            request_params["max_tokens"] = tokens
        
        # Add tools if specified
        if tools:
            request_params["tools"] = tools
            request_params["tool_choice"] = tool_choice
        
        completion = await self.client.chat.completions.create(**request_params)

        self.stats["calls"] += 1
        
        # Extract response
        message = completion.choices[0].message
        result = {
            "answer": message.content or "",
            "rational": ""
        }
        
        # Extract tool calls if present
        if hasattr(message, 'tool_calls') and message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in message.tool_calls
            ]
        
        return result

    async def _qa_with_reasoning(
        self, 
        system_prompt: str, 
        user_prompt: str,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute a query with reasoning enabled."""
        request_params = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": "<think>\n"}
            ],
            "stream": False,
            "temperature": temperature if temperature is not None else self.temperature,
        }
        
        # Add max_tokens if specified
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        if tokens is not None:
            request_params["max_tokens"] = tokens
        
        # Add tools if specified
        if tools:
            request_params["tools"] = tools
            request_params["tool_choice"] = tool_choice
        
        completion = await self.client.chat.completions.create(**request_params)

        self.stats["calls"] += 1
        reasoning = getattr(completion.choices[0].message, "reasoning_content", "")
        
        # Extract response
        message = completion.choices[0].message
        answer = message.content
        
        # Parse JSON if enabled
        if self.parse_json:
            answer = JSONParser.parse(answer)
        
        result = {
            "answer": answer or "",
            "rational": reasoning if reasoning else ""
        }
        
        # Extract tool calls if present
        if hasattr(message, 'tool_calls') and message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in message.tool_calls
            ]
        
        return result

    async def _qa_multimodal(
        self, 
        system_prompt: str, 
        user_prompt: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute a multimodal query or handle full conversation history.
        
        Args:
            user_prompt: Can be either:
                - Content items: [{"type": "text", ...}, {"type": "image", ...}]
                - Full messages: [{"role": "user", "content": ...}, {"role": "assistant", ...}, ...]
        """
        # Check if user_prompt is full message history (has "role" key) or content items (has "type" key)
        if user_prompt and isinstance(user_prompt[0], dict) and "role" in user_prompt[0]:
            # Full conversation history provided - use it directly
            messages = [{"role": "system", "content": system_prompt}] + user_prompt
            mm_kwargs = None
        else:
            # Content items - build user message
            processed_content, mm_kwargs = build_multimodal_message(
                user_prompt, 
                model_name=self.model_name
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": processed_content}
            ]

        # Prepare request parameters
        request_params = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "temperature": temperature if temperature is not None else self.temperature,
        }
        
        # Add max_tokens if specified
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        if tokens is not None:
            request_params["max_tokens"] = tokens

        # Add tools if specified
        if tools:
            request_params["tools"] = tools
            request_params["tool_choice"] = tool_choice

        # Add multimodal processor kwargs if present
        if mm_kwargs:
            request_params["extra_body"] = {"mm_processor_kwargs": mm_kwargs}

        completion = await self.client.chat.completions.create(**request_params)

        self.stats["calls"] += 1
        
        # Extract response
        message = completion.choices[0].message
        answer = message.content
        
        # Parse JSON if enabled
        if self.parse_json:
            answer = JSONParser.parse(answer)
        
        result = {
            "answer": answer or "",
            "rational": ""
        }
        
        # Extract tool calls if present
        if hasattr(message, 'tool_calls') and message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in message.tool_calls
            ]
        
        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics for this API instance."""
        return self.stats.copy()