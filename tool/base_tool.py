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
    """Base class for model-based tools with automatic model sharing.
    
    Tools specify a model_id (defined in model_config.py), and the system
    automatically handles loading, caching, and sharing models across tools.
    
    To use:
        1. Set class attribute: model_id = "clip" or ["clip", "dino"] for multiple models
        2. Implement load_model_components() to unpack and set model components
        3. Done! Model sharing is automatic and transparent.
    """
    
    model_id: Optional[Union[str, List[str]]] = None  # Single or multiple model IDs
    device: str = "cpu"
    is_loaded: bool = False

    def __init__(self, cfg: Optional[Dict] = None, use_zh: bool = False):
        super().__init__(cfg, use_zh)
        self.device = "cpu"
        self.is_loaded = False
        self._model_ids = []  # Normalized list of model IDs

    def load_model_components(self, model_data: Any, device: str) -> None:
        """Unpack and set model components automatically.
        
        This method is called automatically by ensure_loaded(). It unpacks model
        components based on the attribute names defined in MODEL_REGISTRY.
        
        You typically don't need to override this unless you have custom requirements.
        
        Args:
            model_data: Single model tuple or dict of models (if model_id is list)
            device: Device the model is loaded on
        """
        from tool.model_config import get_model_attrs
        
        # Handle multiple models
        if isinstance(model_data, dict):
            for mid, data in model_data.items():
                attrs = get_model_attrs(mid)
                if isinstance(data, tuple):
                    for attr, value in zip(attrs, data):
                        setattr(self, attr, value)
                else:
                    setattr(self, attrs[0], data)
        # Handle single model
        else:
            attrs = get_model_attrs(self._model_ids[0])
            if isinstance(model_data, tuple):
                for attr, value in zip(attrs, model_data):
                    setattr(self, attr, value)
            else:
                setattr(self, attrs[0], model_data)
        
        self.device = device
        self.is_loaded = True

    def unload_model(self) -> None:
        """Release model reference (automatic reference counting)."""
        if self.is_loaded and self._model_ids:
            from tool.model_config import release_model
            for mid in self._model_ids:
                release_model(mid, self.device, tool_name=self.name)
            self.is_loaded = False
            self.device = "cpu"

    def ensure_loaded(self, device: Optional[str] = None) -> None:
        """Ensure model is loaded (use cached if available)."""
        if self.is_loaded:
            return
            
        if not self.model_id:
            raise ValueError(f"Tool {self.name} must set 'model_id'")
        
        from tool.model_config import get_model, _model_cache
        
        # Normalize model_id to list
        self._model_ids = [self.model_id] if isinstance(self.model_id, str) else self.model_id
        
        # Try to find cached model first, otherwise load to specified device
        if device is None:
            # Auto-select: prefer cached model's device
            for mid in self._model_ids:
                for cache_key in _model_cache:
                    if cache_key.startswith(f"{mid}:"):
                        device = cache_key.split(":", 1)[1]
                        break
                if device:
                    break
            
            # Fallback: auto-select GPU
            if device is None:
                try:
                    import torch
                    if torch.cuda.is_available():
                        device = "cuda:0"
                    else:
                        device = "cpu"
                except:
                    device = "cpu"
        
        # Load single or multiple models
        if len(self._model_ids) == 1:
            model_data, _ = get_model(self._model_ids[0], device, self.name)
        else:
            model_data = {
                mid: get_model(mid, device, self.name)[0]
                for mid in self._model_ids
            }
        
        self.load_model_components(model_data, device)

    def call(self, params: Union[str, Dict]) -> Any:
        self.ensure_loaded()
        return self._call_impl(params)

    @abstractmethod
    def _call_impl(self, params: Union[str, Dict]) -> Any:
        """Tool implementation."""
        raise NotImplementedError
