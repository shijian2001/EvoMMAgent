"""Data loading and trace formatting utilities."""

import json
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


def load_results(jsonl_path: str) -> List[Dict]:
    """Load JSONL file into a list of dicts."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_trace_index(memory_dir: str) -> Dict[str, Path]:
    """Scan memory/tasks/ once and return {dataset_id: trace_path} mapping."""
    index: Dict[str, Path] = {}
    tasks_dir = Path(memory_dir) / "tasks"
    if not tasks_dir.exists():
        logger.warning("Tasks dir not found: %s", tasks_dir)
        return index
    for trace_file in tasks_dir.glob("*/trace.json"):
        with open(trace_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        did = str(data.get("dataset_id", ""))
        if did:
            index[did] = trace_file
    logger.info("Indexed %d traces from %s", len(index), tasks_dir)
    return index


def load_trace(trace_path: Path) -> Dict:
    """Load a single trace.json."""
    with open(trace_path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_trace(trace: List[Dict]) -> str:
    """Format trace steps (same as retrieval summarize).

    think/answer → content, action → [tool_name], observation → skip.
    """
    steps: List[str] = []
    for s in trace:
        stype = s.get("type", "")
        if stype in ("think", "answer") and s.get("content"):
            steps.append(s["content"])
        elif stype == "action" and "tool" in s:
            steps.append(f"[{s['tool']}]")
        # observation → skip
    if not steps:
        return "N/A"
    return "\n".join(f"{i}. {step}" for i, step in enumerate(steps, 1))


def resolve_image_paths(trace_data: Dict, task_dir: Path) -> List[str]:
    """Resolve input image paths from the task directory (img_0.png, img_1.png, ...)."""
    images = trace_data.get("input", {}).get("images", [])
    return [str(task_dir / f"{img['id']}.png") for img in images if "id" in img]
