#!/usr/bin/env python3
"""State-level retrieval: StatePipeline end-to-end test.

Covers:
  - Build multi-view state bank from synthetic annotated data (real embeddings)
  - StatePipeline.retrieve(elements) with real embeddings
  - experience_top_n control
  - Multimodal embedding path (text + image)

Usage:
    python unit_test/state_level/test_pipeline.py \
        --embedding_model Qwen/Qwen3-VL-Embedding-2B \
        --embedding_base_url http://localhost:8001/v1
"""

import argparse
import asyncio
import base64
import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helpers import CORRECT_TRACES, ok, section
from config import RetrievalConfig
from mm_memory.state_bank import ALL_VIEWS, StateBank, available_views, compose_view
from mm_memory.retrieval.embedder import Embedder
from mm_memory.retrieval.state_pipeline import StatePipeline


# ── Fake annotated trajectory ───────────────────────────────────────────────

FAKE_TRAJECTORIES = {
    "000001": [
        {
            "state_index": 0,
            "action": {
                "thinking": "I need to find the largest car first.",
                "tool": "localize_objects",
                "parameters": {"query": "car"},
                "observation": "Found 3 cars: bbox_1(0.1,0.2,0.3,0.4), bbox_2(0.5,0.1,0.9,0.8)",
            },
            "q_value": 8,
            "experience": "For color identification tasks, first localize the target object to get a closer view.",
        },
        {
            "state_index": 1,
            "action": {
                "thinking": "",
                "tool": "answer",
                "parameters": {"content": "Red"},
                "observation": None,
            },
            "q_value": 9,
            "experience": "When the cropped image clearly shows the answer, respond directly without further tools.",
        },
    ],
    "000002": [
        {
            "state_index": 0,
            "action": {
                "thinking": "I will detect all people.",
                "tool": "localize_objects",
                "parameters": {"query": "person"},
                "observation": "Found 5 persons.",
            },
            "q_value": 8,
            "experience": "For counting tasks, use localize_objects to detect all instances of the target.",
        },
        {
            "state_index": 1,
            "action": {
                "thinking": "",
                "tool": "answer",
                "parameters": {"content": "5"},
                "observation": None,
            },
            "q_value": 9,
            "experience": "After detecting all instances, count the bounding boxes for the final answer.",
        },
    ],
    "000003": [
        {
            "state_index": 0,
            "action": {
                "thinking": "I need depth estimation.",
                "tool": "estimate_object_depth",
                "parameters": {"query": "dog"},
                "observation": "depth=0.3",
            },
            "q_value": 7,
            "experience": "For depth comparison, estimate depth for each object separately then compare values.",
        },
        {
            "state_index": 1,
            "action": {
                "thinking": "Now estimate cat depth.",
                "tool": "estimate_object_depth",
                "parameters": {"query": "cat"},
                "observation": "depth=0.7",
            },
            "q_value": 8,
            "experience": "Lower depth values indicate closer objects. Compare depth values to determine proximity.",
        },
        {
            "state_index": 2,
            "action": {
                "thinking": "",
                "tool": "answer",
                "parameters": {"content": "The dog"},
                "observation": None,
            },
            "q_value": 9,
            "experience": "Compare depth values directly — lower depth means closer to the camera.",
        },
    ],
    "000005": [
        {
            "state_index": 0,
            "action": {
                "thinking": "Let me examine each shape.",
                "tool": "localize_objects",
                "parameters": {"query": "shapes"},
                "observation": "Found 8 shapes.",
            },
            "q_value": 7,
            "experience": "For pattern recognition, first identify all individual elements before analyzing the sequence.",
        },
        {
            "state_index": 1,
            "action": {
                "thinking": "",
                "tool": "answer",
                "parameters": {"content": "Answer: A"},
                "observation": None,
            },
            "q_value": 8,
            "experience": "After examining individual elements, identify the overarching pattern and select the option.",
        },
    ],
}


async def build_test_state_bank(memory_dir: str, embedder: Embedder):
    """Build a multi-view state bank from fake trajectories using real embeddings."""
    state_metas = []
    per_view_indices = {view: [] for view in ALL_VIEWS}
    per_view_payloads = {view: [] for view in ALL_VIEWS}

    for task_id, trajectory in FAKE_TRAJECTORIES.items():
        trace_data = next(
            (t for t in CORRECT_TRACES if t["task_id"] == task_id), None
        )
        if trace_data is None:
            continue

        for entry in trajectory:
            q = entry.get("q_value", 0)
            step = entry["state_index"]
            elements = StateBank.state_to_elements(trace_data, trajectory, step)
            avail = set(available_views(elements))
            full_text = compose_view(elements, "all").get("text", "")
            state_metas.append({
                "task_id": task_id,
                "state": step,
                "experience": entry.get("experience", ""),
                "q_value": q,
                "state_text": full_text,
                "elements": elements,
                "available_views": sorted(list(avail)),
            })

            meta_idx = len(state_metas) - 1
            for view in ALL_VIEWS:
                if view not in avail:
                    continue
                per_view_indices[view].append(meta_idx)
                per_view_payloads[view].append(compose_view(elements, view))

    bank_dir = os.path.join(memory_dir, "state_bank")
    views_dir = os.path.join(bank_dir, "views")
    os.makedirs(views_dir, exist_ok=True)

    for view in ALL_VIEWS:
        payloads = per_view_payloads[view]
        indices = per_view_indices[view]
        if not payloads:
            continue
        sub_emb = await embedder.encode_view_batch(payloads, batch_size=2)
        full_emb = np.zeros((len(state_metas), sub_emb.shape[1]), dtype=np.float32)
        full_mask = np.zeros((len(state_metas),), dtype=bool)
        for row_i, state_i in enumerate(indices):
            full_emb[state_i] = sub_emb[row_i]
            full_mask[state_i] = True
        np.save(os.path.join(views_dir, f"{view}.npy"), full_emb)
        np.save(os.path.join(views_dir, f"{view}_mask.npy"), full_mask)

    with open(os.path.join(bank_dir, "state_meta.json"), "w", encoding="utf-8") as f:
        json.dump(state_metas, f, ensure_ascii=False, indent=2)

    return StateBank(memory_dir)


# ── Tests ───────────────────────────────────────────────────────────────────

async def test_build_and_retrieve(memory_dir: str, embedder: Embedder):
    """Build state bank with real embeddings, then test retrieval."""
    section("1. Build state bank (real embeddings)")

    bank = await build_test_state_bank(memory_dir, embedder)

    expected = sum(len(traj) for traj in FAKE_TRAJECTORIES.values())
    assert len(bank.state_meta) == expected, \
        f"Expected {expected} states, got {len(bank.state_meta)}"
    any_view = next(iter(bank.view_embeddings.values()))
    ok(f"Built state bank: {len(bank.state_meta)} states, dim={any_view.shape[1]}")

    # ── 2a. Basic retrieval (top-1) ──
    section("2. StatePipeline.retrieve(elements)")

    config_a = RetrievalConfig(
        enable=True, mode="state", bank_memory_dir=memory_dir,
        min_q_value=5, min_score=0.01,
        experience_top_n=1,
    )
    pipeline_a = StatePipeline(config=config_a, state_bank=bank, embedder=embedder)
    trace_car = next(t for t in CORRECT_TRACES if t["task_id"] == "000001")
    elements_car = StateBank.state_to_elements(trace_car, FAKE_TRAJECTORIES["000001"], 1)

    exp_a = await pipeline_a.retrieve(elements_car)
    assert isinstance(exp_a, str) and len(exp_a) > 0, "experience should not be empty"
    entries_a = [l for l in exp_a.splitlines() if l.startswith("#")]
    assert len(entries_a) <= 1, f"top-1 should return at most one experience, got {len(entries_a)}"
    ok(f"[a] top-1: \"{exp_a}\"")

    # ── 2b. top-3 retrieval ──
    config_b = RetrievalConfig(
        enable=True, mode="state", bank_memory_dir=memory_dir,
        min_q_value=5, min_score=0.01,
        experience_top_n=3,
    )
    pipeline_b = StatePipeline(config=config_b, state_bank=bank, embedder=embedder)
    trace_count = next(t for t in CORRECT_TRACES if t["task_id"] == "000002")
    elements_count = StateBank.state_to_elements(trace_count, FAKE_TRAJECTORIES["000002"], 1)

    exp_b = await pipeline_b.retrieve(elements_count)
    assert isinstance(exp_b, str) and len(exp_b) > 0
    entries_b = [l for l in exp_b.splitlines() if l.startswith("#")]
    assert len(entries_b) <= 3, f"top-3 should return at most 3 experiences, got {len(entries_b)}"
    ok(f"[b] top-3: {len(entries_b)} experience entries")

    # ── 2c. Empty result (extreme thresholds) ──
    config_c = RetrievalConfig(
        enable=True, mode="state", bank_memory_dir=memory_dir,
        min_q_value=10, min_score=0.99,
        experience_top_n=1,
    )
    pipeline_c = StatePipeline(config=config_c, state_bank=bank, embedder=embedder)

    exp_c = await pipeline_c.retrieve(elements_car)
    assert exp_c == "", "Should return empty string with extreme thresholds"
    ok("[c] Extreme thresholds → empty string")

    # ── 2d. Multimodal embedding path (text + image) ──
    section("3. Embedder multimodal path")
    tiny_png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Zx5kAAAAASUVORK5CYII="
    )
    img_path = os.path.join(memory_dir, "tiny_test.png")
    with open(img_path, "wb") as f:
        f.write(tiny_png_bytes)

    mm_emb = await embedder.encode_view(
        {
            "text": "Question: Which image is similar?",
            "images": [img_path],
        }
    )
    assert mm_emb.shape[0] == 1 and mm_emb.shape[1] > 0, f"Unexpected multimodal shape: {mm_emb.shape}"
    ok(f"[d] multimodal encode_view shape={mm_emb.shape}")


# ── Main ────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_model", required=True)
    parser.add_argument("--embedding_base_url", required=True)
    parser.add_argument("--embedding_api_key", default="dummy")
    args = parser.parse_args()

    memory_dir = tempfile.mkdtemp(prefix="test_state_pipeline_")
    try:
        embedder = Embedder(
            model_name=args.embedding_model,
            base_url=args.embedding_base_url,
            api_key=args.embedding_api_key,
        )

        await test_build_and_retrieve(memory_dir, embedder)

        print("\n🎉 state_level/test_pipeline ALL PASSED\n")
    finally:
        shutil.rmtree(memory_dir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
