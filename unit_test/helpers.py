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
            {"type": "action", "tool": "localize_objects", "properties": {"query": "car"},
             "observation": "Found 3 cars."},
            {"type": "answer", "content": "Red"},
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
            {"type": "action", "tool": "localize_objects", "properties": {"query": "person"},
             "observation": "Found 5 persons."},
            {"type": "answer", "content": "5"},
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
            {"type": "action", "tool": "estimate_object_depth", "properties": {"query": "dog"},
             "observation": "depth=0.3"},
            {"type": "action", "tool": "estimate_object_depth", "properties": {"query": "cat"},
             "observation": "depth=0.7"},
            {"type": "answer", "content": "The dog"},
        ],
        "answer": "The dog",
        "is_correct": True,
    },
    {
        "task_id": "000004",
        "input": {"question": "Is the text in the image written in English?"},
        "sub_task": "ocr",
        "trace": [
            {"type": "action", "tool": "ocr", "properties": {},
             "observation": "Detected: Hello World"},
            {"type": "answer", "content": "Yes"},
        ],
        "answer": "Yes",
        "is_correct": False,  # filtered when filter_correct=True
    },
    # Trace with repeated tools and long multi-line answer (for index text tests)
    {
        "task_id": "000005",
        "input": {
            "question": "During the IQ test, identify the pattern.",
            "images": [{"id": "img_0", "path": "data/eval/image/test.png"}],
        },
        "sub_task": "IQ Test",
        "trace": [
            {"type": "think", "content": "Let me examine each shape."},
            {"type": "action", "tool": "localize_objects", "properties": {"query": "shapes"},
             "observation": "Found 8 shapes."},
            {"type": "action", "tool": "zoom_in", "properties": {"bbox": [0, 0, 0.1, 0.5]},
             "observation": "Star shape."},
            {"type": "action", "tool": "zoom_in", "properties": {"bbox": [0.1, 0, 0.2, 0.5]},
             "observation": "Bowtie shape."},
            {"type": "action", "tool": "zoom_in", "properties": {"bbox": [0.2, 0, 0.3, 0.5]},
             "observation": "Complex shape."},
            {"type": "answer", "content": "The pattern shows decreasing geometric regularity.\nOption A continues the organic progression.\nAnswer: A"},
        ],
        "answer": (
            "The pattern shows decreasing geometric regularity.\n"
            "Option A continues the organic progression.\n"
            "Answer: A"
        ),
        "is_correct": True,
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
    """Write random embeddings + experiences to {memory_dir}/trace_bank/."""
    n = len(CORRECT_TRACES)
    embeddings = np.random.randn(n, dim).astype(np.float32)
    experiences = [f"Experience {i}" for i in range(n)]

    bank_dir = os.path.join(memory_dir, "trace_bank")
    os.makedirs(bank_dir, exist_ok=True)
    np.save(os.path.join(bank_dir, "embeddings.npy"), embeddings)
    with open(os.path.join(bank_dir, "experiences.json"), "w") as f:
        json.dump(experiences, f)


def cleanup(path: str):
    shutil.rmtree(path, ignore_errors=True)


def ok(msg: str):
    print(f"  ✅ {msg}")


def section(title: str):
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")
