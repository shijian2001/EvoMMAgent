#!/usr/bin/env python3
"""State-level retrieval: pure local logic tests — no network required.

Covers:
  - RetrievalConfig mode/min_q_value/experience_top_n defaults
  - convert_trace_to_trajectory (trace → MDP trajectory)
  - StateBank.state_to_text serialization (no truncation, no caption path)
  - StateBank load + cosine search + Q-value filtering
  - SearchExperiencesTool round/not-yet-used/max_epoch behavior
  - Memory.log_state_retrieval non-step logging behavior
  - StateBank missing error handling

Usage:
    python unit_test/state_level/test_local.py
"""

import asyncio
import json
import os
import sys
import tempfile
import shutil
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

# Paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Stub heavy dependencies for lightweight test environments
for _mod_name in (
    "torch", "torchvision", "torchvision.transforms",
    "torchvision.transforms.functional", "transformers",
    "PIL", "PIL.Image",
    "openai", "dotenv", "qwen_vl_utils", "jsonschema",
):
    sys.modules.setdefault(_mod_name, MagicMock())

from helpers import FAKE_TRACES, CORRECT_TRACES, ok, section
from config import RetrievalConfig, Config
from mm_memory.memory import Memory
from mm_memory.state_bank import StateBank
from tool.search_experiences_tool import SearchExperiencesTool


# ── Fake annotated trajectory data ──────────────────────────────────────────

FAKE_TRAJECTORY = [
    {
        "state_index": 0,
        "action": {
            "thinking": "I need to find the largest car first.",
            "tool": "localize_objects",
            "parameters": {"query": "car"},
            "observation": "Found 3 cars: bbox_1(0.1,0.2,0.3,0.4), bbox_2(0.5,0.1,0.9,0.8), bbox_3(0.2,0.3,0.35,0.45)",
        },
        "q_value": 8,
        "experience": "For color identification tasks, first localize the target object.",
    },
    {
        "state_index": 1,
        "action": {
            "thinking": "bbox_2 is the largest. Let me crop it for a closer look.",
            "tool": "crop",
            "parameters": {"image": "img_0", "bbox": [0.5, 0.1, 0.9, 0.8]},
            "observation": "Cropped car region showing a red sedan",
        },
        "q_value": 8,
        "experience": "After localizing the target, crop the region for a clearer view before making visual judgments.",
    },
    {
        "state_index": 2,
        "action": {
            "thinking": "The car is clearly red based on the cropped image.",
            "tool": "answer",
            "parameters": {"content": "Red"},
            "observation": None,
        },
        "q_value": 9,
        "experience": "When the cropped image clearly shows the answer, respond directly without further tools.",
    },
]


def create_synthetic_state_bank(memory_dir: str, dim: int = 8):
    """Write random multi-view bank files to {memory_dir}/state_bank/."""
    metas = []
    for t in CORRECT_TRACES:
        for entry in FAKE_TRAJECTORY:
            metas.append({
                "task_id": t["task_id"],
                "state": entry["state_index"],
                "experience": entry["experience"],
                "q_value": entry["q_value"],
                "state_text": f"state text for {t['task_id']} step {entry['state_index']}",
            })

    embeddings = np.random.randn(len(metas), dim).astype(np.float32)
    mask = np.ones((len(metas),), dtype=bool)

    bank_dir = os.path.join(memory_dir, "state_bank")
    views_dir = os.path.join(bank_dir, "views")
    os.makedirs(views_dir, exist_ok=True)
    np.save(os.path.join(views_dir, "all.npy"), embeddings)
    np.save(os.path.join(views_dir, "all_mask.npy"), mask)
    with open(os.path.join(bank_dir, "state_meta.json"), "w") as f:
        json.dump(metas, f)


def cleanup(path: str):
    shutil.rmtree(path, ignore_errors=True)


def convert_trace_to_trajectory_local(trace_data):
    """Local copy of trace→trajectory conversion for offline tests."""
    raw_steps = trace_data.get("trace", [])
    trajectory = []
    current_thinking = ""

    for step in raw_steps:
        stype = step.get("type", "")
        if stype == "think":
            current_thinking = step.get("content", "")
        elif stype == "action":
            obs = step.get("observation", "")
            if isinstance(obs, dict):
                obs = obs.get("description", str(obs))
            trajectory.append(
                {
                    "state_index": len(trajectory),
                    "action": {
                        "thinking": current_thinking,
                        "tool": step.get("tool", ""),
                        "parameters": step.get("properties", {}),
                        "observation": str(obs),
                    },
                }
            )
            current_thinking = ""
        elif stype == "answer":
            trajectory.append(
                {
                    "state_index": len(trajectory),
                    "action": {
                        "thinking": current_thinking,
                        "tool": "answer",
                        "parameters": {"content": step.get("content", "")},
                        "observation": None,
                    },
                }
            )
            current_thinking = ""
    return trajectory


# ── Tests ───────────────────────────────────────────────────────────────────

def test_config_mode_defaults():
    """Verify mode, min_q_value, experience_top_n defaults in RetrievalConfig."""
    section("1. Config Mode Defaults")

    rc = RetrievalConfig()
    assert rc.mode == "state", f"Expected mode='state', got '{rc.mode}'"
    assert rc.min_q_value == 7, f"Expected min_q_value=7, got {rc.min_q_value}"
    assert rc.experience_top_n == 1, f"Expected experience_top_n=1, got {rc.experience_top_n}"
    assert rc.enable is False, "retrieval should be disabled by default"
    ok("RetrievalConfig: mode='state', min_q_value=7, experience_top_n=1, enable=False")

    cfg = Config()
    assert cfg.retrieval.mode == "state"
    assert cfg.retrieval.experience_top_n == 1
    ok("Config.retrieval integrated with state mode defaults")


def test_convert_trace_to_trajectory():
    """Verify trace → trajectory conversion merges think+action correctly."""
    section("2. convert_trace_to_trajectory")

    # Trace 000001: think → action → answer (no think before answer)
    traj = convert_trace_to_trajectory_local(FAKE_TRACES[0])
    assert len(traj) == 2, f"Expected 2 entries (1 action + 1 answer), got {len(traj)}"
    assert traj[0]["action"]["tool"] == "localize_objects"
    assert traj[0]["action"]["thinking"] == "I need to find the largest car first."
    assert traj[0]["action"]["observation"] == "Found 3 cars.", \
        f"Observation not captured: '{traj[0]['action']['observation']}'"
    assert traj[0]["action"]["parameters"] == {"query": "car"}, \
        f"Parameters not captured: {traj[0]['action']['parameters']}"
    # Verify answer entry — no preceding think, so thinking is ""
    assert traj[1]["action"]["tool"] == "answer"
    assert traj[1]["action"]["parameters"] == {"content": "Red"}
    assert traj[1]["action"]["thinking"] == "", \
        f"Answer should have empty thinking (no think before answer), got: '{traj[1]['action']['thinking']}'"
    assert traj[1]["action"]["observation"] is None
    ok(f"Trace 000001 → {len(traj)} entries (action + answer), answer has empty thinking")

    # Trace 000003: think → action → action → answer (no think before answer)
    traj3 = convert_trace_to_trajectory_local(FAKE_TRACES[2])
    assert len(traj3) == 3, f"Expected 3 entries (2 actions + 1 answer), got {len(traj3)}"
    tools = [e["action"]["tool"] for e in traj3]
    assert tools == ["estimate_object_depth", "estimate_object_depth", "answer"]
    assert traj3[0]["action"]["observation"] == "depth=0.3"
    assert traj3[1]["action"]["observation"] == "depth=0.7"
    assert traj3[1]["action"]["thinking"] == "", "Second action has no preceding think"
    assert traj3[2]["action"]["parameters"] == {"content": "The dog"}
    assert traj3[2]["action"]["thinking"] == "", "Answer has no preceding think"
    ok(f"Trace 000003 → {len(traj3)} entries: {tools}, observations + answer captured")

    # Trace 000005: think → action × 4 → answer (no think before answer)
    traj5 = convert_trace_to_trajectory_local(FAKE_TRACES[4])
    assert len(traj5) == 5, f"Expected 5 entries (4 actions + 1 answer), got {len(traj5)}"
    assert "Let me examine" in traj5[0]["action"]["thinking"]
    assert traj5[0]["action"]["observation"] == "Found 8 shapes."
    assert traj5[1]["action"]["observation"] == "Star shape."
    assert traj5[4]["action"]["tool"] == "answer"
    assert traj5[4]["action"]["thinking"] == "", "Answer has no preceding think"
    assert "Answer: A" in traj5[4]["action"]["parameters"]["content"]
    ok(f"Trace 000005 → {len(traj5)} entries, all observations + answer captured")


def test_state_to_text_no_truncation():
    """Verify state serialization has NO truncation and ignores caption arg."""
    section("3. StateBank.state_to_text (no truncation)")

    trace_data = {
        "input": {"question": "What color is the car?"},
        "sub_task": "color_recognition",
    }

    # s_0: question/task with prefixes, no action history
    s0 = StateBank.state_to_text(trace_data, FAKE_TRAJECTORY, 0)
    assert "Question: What color is the car?" in s0
    assert "Task: color_recognition" in s0
    assert "localize_objects" not in s0, "s_0 should have no action history"
    ok(f"s_0 = '{s0}'")

    # s_1: prefixed question/task + a_0 summary — full observation, not truncated
    s1 = StateBank.state_to_text(trace_data, FAKE_TRAJECTORY, 1)
    assert "Question: What color is the car?" in s1
    assert "Task: color_recognition" in s1
    assert "localize_objects" in s1
    # The full observation should be present, not truncated
    assert "bbox_3(0.2,0.3,0.35,0.45)" in s1, "Observation should NOT be truncated"
    ok(f"s_1 contains full observation (no truncation)")

    # s_2: question + a_0 + a_1 (crop)
    s2 = StateBank.state_to_text(trace_data, FAKE_TRAJECTORY, 2)
    assert "localize_objects" in s2
    assert "crop" in s2
    assert "Cropped car region" in s2
    ok(f"s_2 includes both actions")

    # s_3 (step_index=3, past the answer entry): answer action should be SKIPPED in text
    s3 = StateBank.state_to_text(trace_data, FAKE_TRAJECTORY, 3)
    assert "localize_objects" in s3
    assert "crop" in s3
    assert "answer" not in s3.lower(), "answer action should NOT appear in state text"
    ok(f"s_3 skips answer action in state text")

    # Verify long observation is preserved
    long_obs = "A" * 500
    long_traj = [{
        "state_index": 0,
        "action": {
            "thinking": "test",
            "tool": "some_tool",
            "parameters": {},
            "observation": long_obs,
        }
    }]
    s_long = StateBank.state_to_text(trace_data, long_traj, 1)
    assert long_obs in s_long, "500-char observation should NOT be truncated"
    ok(f"500-char observation preserved in full")

    ok("State serialization contains only state elements (no caption path)")


def test_state_bank_search():
    """Verify StateBank load, cosine search, and Q-value filtering."""
    section("4. StateBank Load & Search")

    memory_dir = tempfile.mkdtemp(prefix="test_state_")
    try:
        create_synthetic_state_bank(memory_dir, dim=8)

        bank = StateBank(memory_dir)
        assert len(bank.state_meta) == len(CORRECT_TRACES) * len(FAKE_TRAJECTORY)
        any_view = next(iter(bank.view_embeddings.values()))
        ok(f"Loaded {len(bank.state_meta)} states, dim={any_view.shape[1]}")

        # Basic search
        query = np.random.randn(1, 8).astype(np.float32)
        results = bank.search(query, top_k=3)
        assert len(results) <= 3
        assert all("retrieval_score" in r for r in results)
        assert all("experience" in r for r in results)
        assert all("q_value" in r for r in results)
        ok(f"Search → {len(results)} results with experience and q_value")

        # Q-value filtering: min_q=9 should only return q=9 states
        results_high_q = bank.search(query, top_k=10, min_q=9)
        assert all(r["q_value"] >= 9 for r in results_high_q)
        ok(f"min_q=9 → {len(results_high_q)} results (all Q>=9)")

        # min_score filtering
        results_high_score = bank.search(query, top_k=10, min_score=0.99)
        assert all(r["retrieval_score"] >= 0.99 for r in results_high_score)
        ok(f"min_score=0.99 → {len(results_high_score)} results")
    finally:
        cleanup(memory_dir)


def test_state_bank_missing():
    """Verify FileNotFoundError when state_bank/ doesn't exist."""
    section("5. StateBank Error Handling")

    memory_dir = tempfile.mkdtemp(prefix="test_state_")
    try:
        raised = False
        try:
            StateBank(memory_dir)
        except FileNotFoundError:
            raised = True
        assert raised, "Should raise FileNotFoundError"
        ok("FileNotFoundError raised when state_bank/ missing")
    finally:
        cleanup(memory_dir)


class _FakeEmbedder:
    async def encode_multimodal(self, text, image_paths=None):
        return np.ones((1, 8), dtype=np.float32)


class _FakeStateBank:
    def search_view(self, view, query_emb, top_k=1, min_score=0.0, min_q=0):
        del query_emb, min_score, min_q
        rows = [
            {
                "experience": f"{view}: prefer tool-based reasoning",
                "source": "correct",
                "retrieval_score": 0.9,
                "task_id": "000001",
                "state": 0,
            },
            {
                "experience": f"{view}: avoid redundant tool calls",
                "source": "incorrect",
                "retrieval_score": 0.8,
                "task_id": "000002",
                "state": 1,
            },
        ]
        return rows[:top_k]


def test_search_experiences_tool_behavior():
    """Verify tool rounds, not-yet-used views, and max_epoch limit."""
    section("6. SearchExperiencesTool Behavior")

    cfg = SimpleNamespace(max_epoch=2, experience_top_n=2, min_score=0.0, min_q_value=0)
    tool = SearchExperiencesTool(
        state_bank=_FakeStateBank(),
        embedder=_FakeEmbedder(),
        retrieval_config=cfg,
    )
    elements = {
        "question": "Which car is larger?",
        "task": "size_comparison",
        "images": [],
        "observations": [],
    }
    tool.reset_state(elements)

    obs1, log1 = asyncio.run(tool.call_async({"view": "question"}))
    assert "Experiences from similar reasoning states" in obs1
    assert len(log1["experiences"]) == 2
    assert log1["view"] == "question"

    obs2, _ = asyncio.run(tool.call_async({"view": "question"}))
    assert "Not-yet-used views" in obs2

    obs3, log3 = asyncio.run(tool.call_async({"view": "task"}))
    assert "Experiences from similar reasoning states" in obs3
    assert len(log3["experiences"]) == 2

    obs4, log4 = asyncio.run(tool.call_async({"view": "question+task+images+observations"}))
    assert "Retrieval limit reached" in obs4
    assert len(log4["experiences"]) == 0
    ok("search_experiences enforces round limit and not-yet-used guidance")


def test_state_retrieval_logging_non_step():
    """Verify retrieval log is inserted before next step and does not consume step ids."""
    section("7. Retrieval Logging (non-step)")

    memory_dir = tempfile.mkdtemp(prefix="test_state_log_")
    try:
        mem = Memory(base_dir=memory_dir)
        mem.start_task("Test question")
        mem.log_state_retrieval(
            next_step=1,
            rounds=[{"round": 1, "view": "question", "experiences": [{"experience": "x"}]}],
        )
        mem.log_think("I should search experiences first.")
        mem.log_action(tool="calculator", properties={"expression": "1+1"}, observation="Calculation result: 2")

        trace = mem.trace_data["trace"]
        assert isinstance(trace[0], dict) and "experience_for_step_1" in trace[0]
        assert trace[1]["step"] == 1 and trace[1]["type"] == "think"
        assert trace[2]["step"] == 2 and trace[2]["type"] == "action"
        ok("experience_for_step_k logged as non-step and preserved before real steps")
    finally:
        cleanup(memory_dir)


if __name__ == "__main__":
    test_config_mode_defaults()
    test_convert_trace_to_trajectory()
    test_state_to_text_no_truncation()
    test_state_bank_search()
    test_state_bank_missing()
    test_search_experiences_tool_behavior()
    test_state_retrieval_logging_non_step()
    print("\n🎉 state_level/test_local ALL PASSED\n")
