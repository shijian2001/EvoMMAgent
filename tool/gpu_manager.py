"""GPU resource management."""

from typing import Dict, List
from threading import Lock

_lock = Lock()
_usage: Dict[int, int] = {}


def _get_gpus() -> List[int]:
    try:
        import torch
        return list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    except ImportError:
        return []


def _get_free_memory(gpu_id: int) -> int:
    try:
        import torch
        total = torch.cuda.get_device_properties(gpu_id).total_memory
        reserved = torch.cuda.memory_reserved(gpu_id)
        return total - reserved
    except:
        return 0


def get_free_gpu() -> str:
    """Get the best available GPU device string."""
    gpus = _get_gpus()
    if not gpus:
        return "cpu"
    
    with _lock:
        best = max(gpus, key=lambda g: _get_free_memory(g) - _usage.get(g, 0) * 512 * 1024 * 1024)
        return f"cuda:{best}"


def acquire_gpu(device: str) -> None:
    """Mark GPU as in use."""
    if device.startswith("cuda:"):
        gpu_id = int(device.split(":")[1])
        with _lock:
            _usage[gpu_id] = _usage.get(gpu_id, 0) + 1


def release_gpu(device: str) -> None:
    """Mark GPU as freed."""
    if device.startswith("cuda:"):
        gpu_id = int(device.split(":")[1])
        with _lock:
            _usage[gpu_id] = max(0, _usage.get(gpu_id, 0) - 1)


def get_gpu_status() -> Dict:
    """Get GPU status."""
    gpus = _get_gpus()
    return {
        "gpus": [
            {"id": g, "free_memory": _get_free_memory(g), "usage": _usage.get(g, 0)}
            for g in gpus
        ]
    }
