"""API module for LLM service integrations.

Provides customizable chat service for various LLM providers including
GPT, Gemini, Qwen, and DeepSeek with retry logic, API key management,
and multimodal (text, image, video) support.
"""

from api.custom_service import CustomizeChatService

__all__ = [
    "CustomizeChatService",
]

