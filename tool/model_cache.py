"""Model caching and sharing system for tools."""

from typing import Dict, Any, Optional, Tuple, List
from threading import Lock
import logging

logger = logging.getLogger(__name__)

# Model cache with reference counting
# Structure: {cache_key: {"objects": {...}, "ref_count": int}}
_model_cache: Dict[str, Dict[str, Any]] = {}
_cache_lock = Lock()


def get_cached_objects(model_id: str, device: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Get all cached objects (model, processor, etc.) if available, increment ref count.
    
    Returns:
        Tuple of (objects_dict, device) where objects_dict contains all cached components
    """
    with _cache_lock:
        if device:
            cache_key = f"{model_id}:{device}"
            if cache_key in _model_cache:
                entry = _model_cache[cache_key]
                entry["ref_count"] += 1
                logger.info(f'✓ Reusing {model_id} on {device} (refs: {entry["ref_count"]})')
                return entry["objects"], device
        else:
            for cache_key, entry in _model_cache.items():
                if cache_key.startswith(f"{model_id}:"):
                    cached_device = cache_key.split(":", 1)[1]
                    entry["ref_count"] += 1
                    logger.info(f'✓ Reusing {model_id} on {cached_device} (refs: {entry["ref_count"]})')
                    return entry["objects"], cached_device
        return None, None


def get_cached_model(model_id: str, device: Optional[str] = None) -> Tuple[Optional[Any], Optional[str]]:
    """Get cached model if available, increment ref count.
    
    Legacy function for backward compatibility. Use get_cached_objects for full support.
    """
    objects, device = get_cached_objects(model_id, device)
    if objects is not None:
        return objects.get("model"), device
    return None, None


def cache_objects(model_id: str, device: str, objects: Dict[str, Any], tool_name: str = "unknown") -> None:
    """Cache multiple objects (model, processor, etc.) for sharing.
    
    Args:
        model_id: Unique identifier for the model
        device: Device where model is loaded
        objects: Dict of objects to cache, e.g. {"model": model, "processor": processor}
        tool_name: Name of the tool caching these objects
    """
    cache_key = f"{model_id}:{device}"
    with _cache_lock:
        if cache_key not in _model_cache:
            logger.info(f'Caching {model_id} on {device} (objects: {list(objects.keys())})')
            _model_cache[cache_key] = {"objects": objects, "ref_count": 1}
        else:
            # Update existing cache with new objects
            _model_cache[cache_key]["objects"].update(objects)
            _model_cache[cache_key]["ref_count"] += 1


def cache_model(model_id: str, device: str, model: Any, tool_name: str = "unknown") -> None:
    """Cache a model for sharing.
    
    Legacy function for backward compatibility. Use cache_objects for full support.
    """
    cache_objects(model_id, device, {"model": model}, tool_name)


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
                # Clean up all cached objects
                if "objects" in entry:
                    entry["objects"].clear()
                import torch
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f'Error unloading: {e}')
            del _model_cache[cache_key]
        else:
            logger.info(f'Released {model_id} (refs: {entry["ref_count"]})')


def preload_tools(tool_bank: Optional[List[str]] = None, devices: Optional[List[str]] = None) -> Dict[str, str]:
    """Preload models for tools, distribute across GPUs.
    
    Now caches all model-related objects (model, processor, image_processor, etc.)
    """
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
    
    # Filter out already cached models
    models_to_load = []
    model_device_map = {}
    
    with _cache_lock:
        for idx, (model_id, tool_cls) in enumerate(sorted(model_to_tool.items())):
            device = devices[idx % len(devices)]
            cache_key = f"{model_id}:{device}"
            
            if cache_key in _model_cache:
                # Already cached, skip loading
                model_device_map[model_id] = device
            else:
                # Need to load
                models_to_load.append((model_id, tool_cls, device))
    
    # Only log and load if there are new models
    if models_to_load:
        logger.info(f"🚀 Preloading {len(models_to_load)} models across {len(devices)} device(s)...")
        
        for model_id, tool_cls, device in models_to_load:
            try:
                temp_tool = tool_cls()
                temp_tool.load_model(device)
                if temp_tool.is_loaded and temp_tool.model is not None:
                    # Auto-discover and cache all model-related objects
                    if hasattr(temp_tool, '_get_cacheable_objects'):
                        objects_to_cache = temp_tool._get_cacheable_objects()
                    else:
                        objects_to_cache = {"model": temp_tool.model}
                    
                    cache_objects(model_id, device, objects_to_cache, tool_name="preload")
                    model_device_map[model_id] = device
                    logger.info(f"  ✓ {model_id:20s} -> {device} (cached: {list(objects_to_cache.keys())})")
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
