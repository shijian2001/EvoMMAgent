"""Model caching and sharing system for tools."""

from typing import Dict, Any, Optional, Tuple, List
from threading import Lock
import logging

logger = logging.getLogger(__name__)

# Model cache with reference counting
_model_cache: Dict[str, Dict[str, Any]] = {}
_cache_lock = Lock()


def get_cached_model(model_id: str, device: Optional[str] = None) -> Tuple[Optional[Any], Optional[str]]:
    """Get cached model if available, increment ref count."""
    with _cache_lock:
        if device:
            cache_key = f"{model_id}:{device}"
            if cache_key in _model_cache:
                entry = _model_cache[cache_key]
                entry["ref_count"] += 1
                logger.info(f'✓ Reusing {model_id} on {device} (refs: {entry["ref_count"]})')
                return entry["model"], device
        else:
            for cache_key, entry in _model_cache.items():
                if cache_key.startswith(f"{model_id}:"):
                    cached_device = cache_key.split(":", 1)[1]
                    entry["ref_count"] += 1
                    logger.info(f'✓ Reusing {model_id} on {cached_device} (refs: {entry["ref_count"]})')
                    return entry["model"], cached_device
        return None, None


def cache_model(model_id: str, device: str, model: Any, tool_name: str = "unknown") -> None:
    """Cache a model for sharing."""
    cache_key = f"{model_id}:{device}"
    with _cache_lock:
        if cache_key not in _model_cache:
            logger.info(f'Caching {model_id} on {device}')
            _model_cache[cache_key] = {"model": model, "ref_count": 1}
        else:
            _model_cache[cache_key]["ref_count"] += 1


def release_model(model_id: str, device: str, tool_name: str = "unknown") -> None:
    """Release model reference, unload when ref_count reaches 0."""
    cache_key = f"{model_id}:{device}"
    with _cache_lock:
        if cache_key not in _model_cache:
            logger.warning(f'Release non-cached model: {cache_key}')
            return
        
        entry = _model_cache[cache_key]
        entry["ref_count"] -= 1
        
        if entry["ref_count"] <= 0:
            logger.info(f'Unloading {model_id} from {device}')
            try:
                del entry["model"]
                import torch
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f'Error unloading: {e}')
            del _model_cache[cache_key]
        else:
            logger.info(f'Released {model_id} (refs: {entry["ref_count"]})')


def preload_tools(tool_bank: Optional[List[str]] = None, devices: Optional[List[str]] = None) -> Dict[str, str]:
    """Preload models for tools, distribute across GPUs."""
    from tool.base_tool import TOOL_REGISTRY
    
    if tool_bank is None:
        tools_to_load = TOOL_REGISTRY
    else:
        tools_to_load = {name: TOOL_REGISTRY[name] for name in tool_bank if name in TOOL_REGISTRY}
        missing = [name for name in tool_bank if name not in TOOL_REGISTRY]
        if missing:
            logger.warning(f"Tools not found: {missing}")
    
    model_to_tool = {}
    for tool_name, tool_cls in tools_to_load.items():
        if hasattr(tool_cls, 'model_id') and tool_cls.model_id:
            model_to_tool[tool_cls.model_id] = tool_cls
    
    if not model_to_tool:
        logger.info("No models to preload")
        return {}
    
    if devices is None:
        import torch
        if torch.cuda.is_available():
            devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        else:
            devices = ["cpu"]
    
    logger.info(f"🚀 Preloading {len(model_to_tool)} models across {len(devices)} device(s)...")
    
    model_device_map = {}
    for idx, (model_id, tool_cls) in enumerate(sorted(model_to_tool.items())):
        device = devices[idx % len(devices)]
        try:
            temp_tool = tool_cls()
            temp_tool.load_model(device)
            if temp_tool.is_loaded and temp_tool.model is not None:
                cache_model(model_id, device, temp_tool.model, tool_name="preload")
                model_device_map[model_id] = device
                logger.info(f"  ✓ {model_id:20s} -> {device}")
            else:
                logger.error(f"  ✗ {model_id:20s} -> {device}")
        except Exception as e:
            logger.error(f"  ✗ {model_id:20s} -> {device} ({e})")
    
    return model_device_map


def get_cache_stats() -> Dict[str, Any]:
    """Get model cache statistics."""
    with _cache_lock:
        return {
            "cached_models": len(_model_cache),
            "models": [
                {
                    "key": key,
                    "ref_count": entry["ref_count"],
                }
                for key, entry in _model_cache.items()
            ]
        }
