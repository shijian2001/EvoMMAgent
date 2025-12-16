"""Base classes for tools."""

import json
import asyncio
import jsonschema
from abc import ABC, abstractmethod
from typing import Union, Dict, Any, Optional, List

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
    """Base class for model-based tools with automatic caching and sharing.
    
    Set model_id, implement load_model() to set self.model, implement _call_impl().
    """
    
    model_id: Optional[str] = None
    model: Any = None
    device: str = "cpu"
    is_loaded: bool = False

    def __init__(self, cfg: Optional[Dict] = None, use_zh: bool = False):
        super().__init__(cfg, use_zh)
        self.device = "cpu"
        self.is_loaded = False

    def load_model(self, device: str) -> None:
        """Load model and set to instance attributes.
        
        Override this method to define how to create and set your model.
        
        Example:
            def load_model(self, device):
                import easyocr
                self.reader = easyocr.Reader(["en"], gpu=device.startswith("cuda"))
                self.device = device
                self.is_loaded = True
        
        Args:
            device: Target device (e.g., "cuda:0", "cpu")
        """
        raise NotImplementedError(f"Tool {self.name} must implement load_model()")

    def unload_model(self) -> None:
        """Release model reference."""
        if self.is_loaded and self.model_id:
            from tool.model_cache import release_model
            release_model(self.model_id, self.device, tool_name=self.name)
            self.is_loaded = False
            self.device = "cpu"

    def ensure_loaded(self, device: Optional[str] = None) -> None:
        """Load model from cache or create new one."""
        if self.is_loaded:
            return
        if not self.model_id:
            raise ValueError(f"Tool {self.name} must set 'model_id'")
            
        from tool.model_cache import get_cached_model, cache_model
        
        cached_model, cached_device = get_cached_model(self.model_id, device)
        if cached_model is not None:
            self.model = cached_model
            self.device = cached_device
            self.is_loaded = True
        else:
            if device is None:
                try:
                    import torch
                    device = "cuda:0" if torch.cuda.is_available() else "cpu"
                except:
                    device = "cpu"
            self.load_model(device)
            if self.is_loaded and self.model is not None:
                cache_model(self.model_id, self.device, self.model, self.name)

    def call(self, params: Union[str, Dict]) -> Any:
        self.ensure_loaded()
        return self._call_impl(params)

    @abstractmethod
    def _call_impl(self, params: Union[str, Dict]) -> Any:
        """Implement tool logic here."""
        raise NotImplementedError
