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
        temperature: float = 0.2,
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
        system: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        rational: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Send messages to the model with system prompt.

        Args:
            system: System prompt
            messages: Conversation messages in OpenAI format:
                [
                    {"role": "user", "content": "..." or [{"type": "text", ...}, {"type": "image", ...}]},
                    {"role": "assistant", "content": "...", "tool_calls": [...]},
                    {"role": "tool", "tool_call_id": "...", "content": "..."}
                ]
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
            
        Example:
            # Single turn
            response = await qa(
                system="You are a helpful assistant.",
                messages=[{"role": "user", "content": "Hello"}]
            )
            
            # Multi-turn with tools
            response = await qa(
                system="You are a helpful assistant.",
                messages=[
                    {"role": "user", "content": [{"type": "text", "text": "..."}, {"type": "image", "image": "..."}]},
                    {"role": "assistant", "tool_calls": [...]},
                    {"role": "tool", "content": "..."}
                ]
            )
        """
        start_time = time.time()
        
        if rational and self.model_name.lower() not in [m.lower() for m in self.SUPPORTED_REASONING_MODELS]:
            raise ValueError(f"Model {self.model_name} does not support reasoning")

        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                # Process messages with multimodal content
                result = await self._qa(
                    system=system,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    rational=rational,
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

    async def _qa(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        rational: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Internal method to execute query with OpenAI-format messages.
        
        Processes multimodal content and handles tool calls.
        
        Args:
            system: System prompt
            messages: Conversation messages in OpenAI format
            tools: Tool definitions
            tool_choice: Tool choice mode
            rational: Whether to enable reasoning
            temperature: Temperature
            max_tokens: Max tokens
            
        Returns:
            Dict with answer and optional tool_calls
        """
        # Process each message's multimodal content (all roles, not just user)
        processed_messages = []
        mm_kwargs = None
        
        for msg in messages:
            # Process any message with list content (could be user, tool, etc.)
            if isinstance(msg.get("content"), list):
                # Convert multimodal content to OpenAI format
                processed_content, temp_mm_kwargs = build_multimodal_message(
                    msg["content"],
                    model_name=self.model_name
                )
                processed_messages.append({
                    **msg,  # Preserve role, tool_call_id, and other fields
                    "content": processed_content
                })
                # Use mm_kwargs from the last message with multimodal content
                if temp_mm_kwargs:
                    mm_kwargs = temp_mm_kwargs
            else:
                # Keep text-only messages as-is
                processed_messages.append(msg)
        
        # Build full message list with system prompt
        full_messages = [
            {"role": "system", "content": system},
            *processed_messages
        ]
        
        # Add reasoning prompt if enabled
        if rational:
            full_messages.append({"role": "assistant", "content": "<think>\n"})

        # Prepare request parameters
        request_params = {
            "model": self.model_name,
            "messages": full_messages,
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
        
        # Build extra_body with all custom params
        extra_body = {}
        
        # Add multimodal processor kwargs if present (only for Qwen models)
        if mm_kwargs:
            extra_body["mm_processor_kwargs"] = mm_kwargs
        
        # Add logit_bias to vLLM sampling params when using tools
        if tools:
            # Prevent tool call format errors by discouraging specific tokens
            extra_body["logit_bias"] = {
                147926: -100,
                151478: -100,
                30543: -100
            }
        
        # Apply extra_body if not empty
        if extra_body:
            request_params["extra_body"] = extra_body

        # Call API
        completion = await self.client.chat.completions.create(**request_params)

        self.stats["calls"] += 1
        
        # Extract response
        message = completion.choices[0].message
        answer = message.content
        
        # Parse JSON if enabled
        if self.parse_json and answer:
            answer = JSONParser.parse(answer)
        
        # Extract reasoning if available
        reasoning = getattr(message, "reasoning_content", "") if rational else ""
        
        result = {
            "answer": answer or "",
            "rational": reasoning
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