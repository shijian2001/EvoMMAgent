"""Shared fixtures and utilities for retrieval tests."""

import json
import os
import sys
import tempfile
import shutil

import numpy as np

# Project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Fake trace data ──────────────────────────────────────────────────────────

FAKE_TRACES = [
    {
        "task_id": "000001",
        "input": {"question": "What color is the largest car in the image?"},
        "sub_task": "color_recognition",
        "trace": [
            {"type": "think", "content": "I need to find the largest car first."},
            {"type": "action", "tool": "localize_objects", "args": {"query": "car"}},
            {"type": "observation", "content": "Found 3 cars."},
            {"type": "think", "content": "The largest bounding box is the red car."},
        ],
        "answer": "Red",
        "is_correct": True,
    },
    {
        "task_id": "000002",
        "input": {"question": "How many people are in the photo?"},
        "sub_task": "counting",
        "trace": [
            {"type": "think", "content": "I will detect all people."},
            {"type": "action", "tool": "localize_objects", "args": {"query": "person"}},
            {"type": "observation", "content": "Found 5 persons."},
        ],
        "answer": "5",
        "is_correct": True,
    },
    {
        "task_id": "000003",
        "input": {"question": "Which object is closer, the dog or the cat?"},
        "sub_task": "depth_estimation",
        "trace": [
            {"type": "think", "content": "I need depth estimation."},
            {"type": "action", "tool": "estimate_object_depth", "args": {"query": "dog"}},
            {"type": "observation", "content": "depth=0.3"},
            {"type": "action", "tool": "estimate_object_depth", "args": {"query": "cat"}},
            {"type": "observation", "content": "depth=0.7"},
        ],
        "answer": "The dog",
        "is_correct": True,
    },
    {
        "task_id": "000004",
        "input": {"question": "Is the text in the image written in English?"},
        "sub_task": "ocr",
        "trace": [
            {"type": "action", "tool": "ocr", "args": {}},
            {"type": "observation", "content": "Detected: Hello World"},
        ],
        "answer": "Yes",
        "is_correct": False,  # filtered when filter_correct=True
    },
]

CORRECT_TRACES = [t for t in FAKE_TRACES if t.get("is_correct", False)]


# ── Helpers ──────────────────────────────────────────────────────────────────

def create_fake_memory_dir() -> str:
    """Create a temp dir with fake trace.json files. Returns path."""
    tmp = tempfile.mkdtemp(prefix="test_retrieval_")
    for t in FAKE_TRACES:
        task_dir = os.path.join(tmp, "tasks", t["task_id"])
        os.makedirs(task_dir, exist_ok=True)
        with open(os.path.join(task_dir, "trace.json"), "w") as f:
            json.dump(t, f)
    return tmp


def create_synthetic_bank(memory_dir: str, dim: int = 8):
    """Write random embeddings + task_ids to {memory_dir}/bank/."""
    task_ids = [t["task_id"] for t in CORRECT_TRACES]
    embeddings = np.random.randn(len(task_ids), dim).astype(np.float32)

    bank_dir = os.path.join(memory_dir, "bank")
    os.makedirs(bank_dir, exist_ok=True)
    np.save(os.path.join(bank_dir, "embeddings.npy"), embeddings)
    with open(os.path.join(bank_dir, "task_ids.json"), "w") as f:
        json.dump(task_ids, f)


def cleanup(path: str):
    shutil.rmtree(path, ignore_errors=True)


def ok(msg: str):
    print(f"  ✅ {msg}")


def section(title: str):
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")
