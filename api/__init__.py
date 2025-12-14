"""
Asynchronous API package for LLM interactions.

This package provides:
- QAWrapper: Simple async wrapper for single API key
- APIPool: Load-balanced pool with multiple API keys and automatic load balancing
- Vision utilities: Multimodal (image/video) message processing
- Key loader: Load API keys from environment
"""

from .wrapper import QAWrapper
from .async_pool import APIPool
from .vision_utils import build_multimodal_message, build_video_message, build_image_message
from .json_parser import JSONParser
from .utils import load_api_keys

__all__ = [
    "QAWrapper",
    "APIPool", 
    "build_multimodal_message",
    "build_video_message",
    "build_image_message",
    "JSONParser",
    "load_api_keys",
]

__version__ = "2.0.0"
