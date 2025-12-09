"""Base classes for tools."""

import json
import asyncio
import jsonschema
from abc import ABC, abstractmethod
from typing import Union, Dict, Any, Optional

TOOL_REGISTRY: Dict[str, type] = {}


def register_tool(name: str):
    """Decorator to register a tool class."""
    def decorator(cls):
        cls.name = name
        TOOL_REGISTRY[name] = cls
        return cls
    return decorator


class BasicTool(ABC):
    """Base class for tools."""
    
    name: str = ""
    description_en: str = ""
    description_zh: str = ""
    parameters: Dict = {}
    example: str = ""

    def __init__(self, cfg: Optional[Dict] = None, use_zh: bool = False):
        cfg = cfg or {}
        self.tool_name = cfg.get("name") or self.name
        self.name_for_model = cfg.get("name_for_model") or self.name
        self.name_for_human = cfg.get("name_for_human") or self.name
        self.tool_description = cfg.get(f"description_{'zh' if use_zh else 'en'}") or \
                                (self.description_zh if use_zh else self.description_en)
        self.tool_example = cfg.get("example") or self.example
        if cfg.get("parameters"):
            self.parameters = cfg["parameters"]

    @abstractmethod
    def call(self, params: Union[str, Dict]) -> Any:
        """Execute the tool."""
        raise NotImplementedError

    async def call_async(self, params: Union[str, Dict]) -> Any:
        """Async version."""
        return await asyncio.to_thread(self.call, params)

    def parse_params(self, params: Union[str, Dict]) -> Dict:
        """Parse and validate parameters."""
        if isinstance(params, str):
            params = params.strip()
            if params.startswith('```'):
                params = '\n'.join(params.split('\n')[1:-1])
            params = json.loads(params)
        
        if self.parameters:
            jsonschema.validate(instance=params, schema=self.parameters)
        return params


class ModelBasedTool(BasicTool):
    """Base class for model-based tools with lazy GPU loading."""
    
    model: Any = None
    device: str = "cpu"
    is_loaded: bool = False

    def __init__(self, cfg: Optional[Dict] = None, use_zh: bool = False):
        super().__init__(cfg, use_zh)
        self.model = None
        self.device = "cpu"
        self.is_loaded = False

    @abstractmethod
    def load_model(self, device: str) -> None:
        """Load model to device."""
        raise NotImplementedError

    def unload_model(self) -> None:
        """Unload model and free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            self.is_loaded = False
            self.device = "cpu"
            try:
                import torch
                torch.cuda.empty_cache()
            except:
                pass

    def ensure_loaded(self, device: Optional[str] = None) -> None:
        """Ensure model is loaded."""
        if not self.is_loaded:
            from tool.gpu_manager import get_free_gpu
            self.load_model(device or get_free_gpu())

    def call(self, params: Union[str, Dict]) -> Any:
        self.ensure_loaded()
        return self._call_impl(params)

    @abstractmethod
    def _call_impl(self, params: Union[str, Dict]) -> Any:
        """Tool implementation."""
        raise NotImplementedError
