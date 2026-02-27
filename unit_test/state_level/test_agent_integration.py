#!/usr/bin/env python3
"""Integration tests: verify search_experiences tool behavior inside the agent loop.

Covers:
  - search_experiences calls do NOT increment action_turns (only total_turns)
  - Retrieval logs are flushed as experience_for_step_k before real steps
  - reset_state is called on trajectory (state) transitions
  - max_epoch is enforced inside the agent loop
  - search_experiences appears in tool_bank and tools schema
  - max_total_turns safety stop works
  - Flush happens before answer when no real tool calls precede it

Usage:
    python unit_test/state_level/test_agent_integration.py
"""

import asyncio
import importlib
import json
import os
import sys
import tempfile
import shutil
from types import SimpleNamespace, ModuleType
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Stub out heavy dependencies so the test runs in lightweight environments
# (no torch, openai, etc.).  setdefault keeps real modules if already loaded.
for _mod_name in (
    "torch", "torchvision", "torchvision.transforms",
    "torchvision.transforms.functional", "transformers",
    "PIL", "PIL.Image",
    "openai", "dotenv", "qwen_vl_utils", "jsonschema",
):
    sys.modules.setdefault(_mod_name, MagicMock())

from config import Config, RetrievalConfig, AgentConfig
from mm_memory.memory import Memory
from mm_memory.state_bank import StateBank, ALL_VIEWS
from tool.search_experiences_tool import SearchExperiencesTool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ok(msg: str):
    print(f"  ✅ {msg}")


def section(title: str):
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


_CALL_COUNTER = 0


def _make_tool_call(tool_name: str, arguments: dict, call_id: str = "") -> dict:
    global _CALL_COUNTER
    _CALL_COUNTER += 1
    return {
        "id": call_id or f"call_{_CALL_COUNTER:04d}",
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": json.dumps(arguments),
        },
    }


def _llm_tool_response(tool_name: str, arguments: dict, thinking: str = "") -> dict:
    return {
        "answer": thinking,
        "tool_calls": [_make_tool_call(tool_name, arguments)],
    }


def _llm_answer_response(answer: str) -> dict:
    return {"answer": answer, "tool_calls": None}


class _FakeEmbedder:
    async def encode_view(self, composed):
        return np.ones((1, 8), dtype=np.float32)


class _FakeStateBank:
    """Deterministic state bank returning view-tagged experiences."""

    def __init__(self):
        self.state_meta = [{"task_id": "T001", "state": 0}]
        self.view_embeddings = {"all": np.ones((1, 8))}

    def search_view(self, view, query_emb, top_k=1, min_score=0.0, min_q=0):
        return [
            {
                "experience": f"[{view}] Use careful reasoning.",
                "source": "correct",
                "retrieval_score": 0.85,
                "task_id": "T001",
                "state": 0,
            }
        ] * top_k

    @staticmethod
    def state_to_elements(trace_data, trajectory, state_index):
        return {
            "question": trace_data.get("input", {}).get("question", ""),
            "task": trace_data.get("sub_task", ""),
            "images": [],
            "observations": [
                t["action"]["observation"]
                for t in trajectory[:state_index]
                if t["action"].get("observation")
            ],
        }


class _FakeTool:
    """Minimal tool that returns a text observation."""
    name = "fake_tool"
    name_for_model = "fake_tool"
    name_for_human = "fake_tool"
    tool_description = "A fake tool."
    tool_example = ""
    parameters = {"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]}

    def call(self, params, **kwargs):
        if isinstance(params, str):
            params = json.loads(params)
        return f"fake_tool result for {params.get('input', '')}"


def _build_agent_skeleton(
    memory_dir: str,
    max_iterations: int = 5,
    max_epoch: int = 2,
    experience_top_n: int = 1,
):
    """Build a minimal MultimodalAgent-like object for testing the act() loop.

    Instead of calling the real __init__ (which needs API keys, templates, GPU tools),
    we construct just enough state for act() to work by patching internals.
    """
    from agent.mm_agent import MultimodalAgent

    cfg = Config(
        agent=AgentConfig(max_iterations=max_iterations, enable_memory=True, memory_dir=memory_dir),
        retrieval=RetrievalConfig(
            enable=True,
            mode="state",
            bank_memory_dir=memory_dir,
            experience_top_n=experience_top_n,
            max_epoch=max_epoch,
            min_score=0.0,
            min_q_value=0,
        ),
    )

    agent = object.__new__(MultimodalAgent)

    agent.name = "TestAgent"
    agent.description = "test"
    agent.use_zh = False
    agent.model_name = "test-model"
    agent.max_iterations = max_iterations
    agent.temperature = 0.0
    agent.max_tokens = None
    agent.enable_memory = True
    agent.memory_dir = memory_dir
    agent.config = cfg
    agent.special_answer_token = "\nAnswer:"

    agent.tool_bank = {}
    agent.tool_bank["fake_tool"] = _FakeTool()

    search_tool = SearchExperiencesTool(
        state_bank=_FakeStateBank(),
        embedder=_FakeEmbedder(),
        retrieval_config=cfg.retrieval,
    )
    agent.search_experiences_tool = search_tool
    agent.tool_bank["search_experiences"] = search_tool

    agent.retrieval_pipeline = None

    agent.api_pool = AsyncMock()

    import jinja2
    template_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "template")
    agent.jinja_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_dir),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    agent.mm_agent_template_file = "MMAgent_EN.jinja2"

    return agent


def _run(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_search_does_not_count_as_action_turn():
    """search_experiences should NOT increment action_turns; only real tools do."""
    section("1. search_experiences does not count as action turn")

    memory_dir = tempfile.mkdtemp(prefix="test_int_")
    try:
        agent = _build_agent_skeleton(memory_dir, max_iterations=5, max_epoch=2, experience_top_n=1)

        responses = iter([
            _llm_tool_response("search_experiences", {"view": "question"}),
            _llm_tool_response("search_experiences", {"view": "task"}),
            _llm_tool_response("fake_tool", {"input": "x"}, thinking="I should use fake_tool."),
            _llm_answer_response("The answer is 42."),
        ])
        agent.api_pool.execute = AsyncMock(side_effect=lambda *a, **kw: next(responses))

        result = _run(agent.act("What is 6*7?", verbose=False, return_history=True))

        assert result["success"] is True, f"Expected success, got {result}"
        assert result["response"] == "The answer is 42."

        history = result["history"]
        search_calls = [h for h in history if h.get("action") == "search_experiences"]
        real_calls = [h for h in history if h.get("action") == "fake_tool"]
        assert len(search_calls) == 2, f"Expected 2 search calls, got {len(search_calls)}"
        assert len(real_calls) == 1, f"Expected 1 real tool call, got {len(real_calls)}"

        task_dirs = [d for d in os.listdir(os.path.join(memory_dir, "tasks")) if os.path.isdir(os.path.join(memory_dir, "tasks", d))]
        assert len(task_dirs) == 1
        trace_path = os.path.join(memory_dir, "tasks", task_dirs[0], "trace.json")
        with open(trace_path) as f:
            trace_data = json.load(f)
        trace = trace_data["trace"]

        exp_entries = [e for e in trace if any(k.startswith("experience_for_step_") for k in e)]
        step_entries = [e for e in trace if "step" in e]

        assert len(exp_entries) >= 1, f"Expected ≥1 experience entries, got {len(exp_entries)}"

        step_numbers = [e["step"] for e in step_entries]
        for i in range(1, len(step_numbers)):
            assert step_numbers[i] == step_numbers[i - 1] + 1, (
                f"Step numbers not consecutive: {step_numbers}"
            )

        ok(f"history: {len(search_calls)} search + {len(real_calls)} real; "
           f"trace: {len(exp_entries)} exp entries, steps={step_numbers}")
    finally:
        shutil.rmtree(memory_dir, ignore_errors=True)


def test_retrieval_logs_interleaved():
    """Retrieval logs appear before the corresponding real step in trace."""
    section("2. Retrieval logs interleaved correctly")

    memory_dir = tempfile.mkdtemp(prefix="test_int_")
    try:
        agent = _build_agent_skeleton(memory_dir, max_iterations=5, max_epoch=2, experience_top_n=1)

        responses = iter([
            _llm_tool_response("search_experiences", {"view": "question"}),
            _llm_tool_response("fake_tool", {"input": "step1"}, thinking="Plan step 1."),
            _llm_tool_response("search_experiences", {"view": "task"}),
            _llm_tool_response("fake_tool", {"input": "step2"}, thinking="Plan step 2."),
            _llm_answer_response("Done."),
        ])
        agent.api_pool.execute = AsyncMock(side_effect=lambda *a, **kw: next(responses))

        result = _run(agent.act("Test interleave", verbose=False, return_history=True))
        assert result["success"] is True

        task_dirs = os.listdir(os.path.join(memory_dir, "tasks"))
        trace_path = os.path.join(memory_dir, "tasks", task_dirs[0], "trace.json")
        with open(trace_path) as f:
            trace = json.load(f)["trace"]

        entry_types = []
        for e in trace:
            if "step" in e:
                entry_types.append(e["type"])
            else:
                key = next(k for k in e if k.startswith("experience_for_step_"))
                entry_types.append(key)

        assert "experience_for_step_1" in entry_types, f"Missing experience_for_step_1 in {entry_types}"
        exp1_idx = entry_types.index("experience_for_step_1")
        think1_idx = entry_types.index("think")
        assert exp1_idx < think1_idx, (
            f"experience_for_step_1 (idx={exp1_idx}) should precede first think (idx={think1_idx})"
        )

        ok(f"Trace entry order: {entry_types}")
    finally:
        shutil.rmtree(memory_dir, ignore_errors=True)


def test_reset_state_on_trajectory_change():
    """After a real tool call advances the trajectory, reset_state is called
    so previously used views become available again."""
    section("3. reset_state on trajectory change")

    memory_dir = tempfile.mkdtemp(prefix="test_int_")
    try:
        agent = _build_agent_skeleton(memory_dir, max_iterations=5, max_epoch=2, experience_top_n=1)
        original_reset = agent.search_experiences_tool.reset_state
        reset_calls = []

        def tracking_reset(elements):
            reset_calls.append(elements)
            return original_reset(elements)

        agent.search_experiences_tool.reset_state = tracking_reset

        responses = iter([
            _llm_tool_response("search_experiences", {"view": "question"}),
            _llm_tool_response("fake_tool", {"input": "s1"}, thinking="Step 1."),
            _llm_tool_response("search_experiences", {"view": "question"}),
            _llm_answer_response("Answer."),
        ])
        agent.api_pool.execute = AsyncMock(side_effect=lambda *a, **kw: next(responses))

        result = _run(agent.act("Test reset", verbose=False, return_history=True))
        assert result["success"] is True

        assert len(reset_calls) >= 2, (
            f"Expected ≥2 reset_state calls (initial + after state change), got {len(reset_calls)}"
        )

        search_calls = [h for h in result["history"] if h.get("action") == "search_experiences"]
        assert len(search_calls) == 2
        for sc in search_calls:
            assert "Round 1/" in sc["observation"], (
                f"Second search_experiences('question') should succeed after reset, got: {sc['observation']}"
            )

        ok(f"reset_state called {len(reset_calls)} times; both search('question') returned Round 1")
    finally:
        shutil.rmtree(memory_dir, ignore_errors=True)


def test_max_epoch_enforced_in_loop():
    """After max_epoch retrieval rounds in the same state, further calls return limit message."""
    section("4. max_epoch enforced inside agent loop")

    memory_dir = tempfile.mkdtemp(prefix="test_int_")
    try:
        agent = _build_agent_skeleton(memory_dir, max_iterations=5, max_epoch=1, experience_top_n=1)

        responses = iter([
            _llm_tool_response("search_experiences", {"view": "question"}),
            _llm_tool_response("search_experiences", {"view": "task"}),
            _llm_answer_response("Final."),
        ])
        agent.api_pool.execute = AsyncMock(side_effect=lambda *a, **kw: next(responses))

        result = _run(agent.act("Test epoch", verbose=False, return_history=True))
        assert result["success"] is True

        search_obs = [
            h["observation"] for h in result["history"] if h.get("action") == "search_experiences"
        ]
        assert len(search_obs) == 2
        assert "Round 1/1" in search_obs[0]
        assert "Retrieval limit reached" in search_obs[1]

        ok(f"1st search: Round 1/1; 2nd search: limit reached")
    finally:
        shutil.rmtree(memory_dir, ignore_errors=True)


def test_tool_bank_and_schema():
    """search_experiences should be in tool_bank and the generated tools schema."""
    section("5. tool_bank and tools schema")

    memory_dir = tempfile.mkdtemp(prefix="test_int_")
    try:
        agent = _build_agent_skeleton(memory_dir)

        assert "search_experiences" in agent.tool_bank
        schema = agent._build_tools_schema()
        se_schemas = [s for s in schema if s["function"]["name"] == "search_experiences"]
        assert len(se_schemas) == 1, f"Expected 1 search_experiences schema, got {len(se_schemas)}"

        se = se_schemas[0]["function"]
        assert "view" in se["parameters"]["properties"]
        assert se["parameters"]["properties"]["view"]["enum"] == ALL_VIEWS

        ok(f"search_experiences in tool_bank; schema has view enum with {len(ALL_VIEWS)} views")
    finally:
        shutil.rmtree(memory_dir, ignore_errors=True)


def test_max_total_turns_safety():
    """When LLM only calls search_experiences forever, max_total_turns stops the loop."""
    section("6. max_total_turns safety stop")

    memory_dir = tempfile.mkdtemp(prefix="test_int_")
    try:
        agent = _build_agent_skeleton(memory_dir, max_iterations=2, max_epoch=3, experience_top_n=1)

        infinite_search = _llm_tool_response("search_experiences", {"view": "question"})

        def always_search(*a, **kw):
            return {
                "answer": "",
                "tool_calls": [_make_tool_call("search_experiences", {"view": "question"})],
            }

        agent.api_pool.execute = AsyncMock(side_effect=always_search)

        result = _run(agent.act("Test safety", verbose=False, return_history=True))

        assert result["success"] is False, "Should fail due to safety stop"
        assert "Safety stop" in result["response"] or "Maximum" in result["response"]
        ok(f"Safety stop triggered: {result['response'][:80]}")
    finally:
        shutil.rmtree(memory_dir, ignore_errors=True)


def test_flush_before_answer_no_real_tool():
    """When search_experiences is followed directly by answer (no real tool),
    retrieval logs should still be flushed before the answer step."""
    section("7. Flush retrieval logs before answer (no real tool)")

    memory_dir = tempfile.mkdtemp(prefix="test_int_")
    try:
        agent = _build_agent_skeleton(memory_dir, max_iterations=5, max_epoch=2, experience_top_n=1)

        responses = iter([
            _llm_tool_response("search_experiences", {"view": "question"}),
            _llm_answer_response("Direct answer."),
        ])
        agent.api_pool.execute = AsyncMock(side_effect=lambda *a, **kw: next(responses))

        result = _run(agent.act("Test direct answer", verbose=False, return_history=True))
        assert result["success"] is True
        assert result["response"] == "Direct answer."

        task_dirs = os.listdir(os.path.join(memory_dir, "tasks"))
        trace_path = os.path.join(memory_dir, "tasks", task_dirs[0], "trace.json")
        with open(trace_path) as f:
            trace = json.load(f)["trace"]

        exp_entries = [e for e in trace if any(k.startswith("experience_for_step_") for k in e)]
        answer_entries = [e for e in trace if e.get("type") == "answer"]

        assert len(exp_entries) >= 1, f"Expected ≥1 experience entry, got {len(exp_entries)}"
        assert len(answer_entries) == 1

        exp_idx = trace.index(exp_entries[0])
        ans_idx = trace.index(answer_entries[0])
        assert exp_idx < ans_idx, (
            f"Experience entry (idx={exp_idx}) should precede answer (idx={ans_idx})"
        )

        ok(f"Retrieval log at idx={exp_idx}, answer at idx={ans_idx}")
    finally:
        shutil.rmtree(memory_dir, ignore_errors=True)


def test_experience_top_n_per_round():
    """Each retrieval round should return experience_top_n experiences."""
    section("8. experience_top_n per round")

    memory_dir = tempfile.mkdtemp(prefix="test_int_")
    try:
        top_n = 3
        agent = _build_agent_skeleton(memory_dir, max_iterations=5, max_epoch=2, experience_top_n=top_n)

        responses = iter([
            _llm_tool_response("search_experiences", {"view": "question"}),
            _llm_answer_response("Done."),
        ])
        agent.api_pool.execute = AsyncMock(side_effect=lambda *a, **kw: next(responses))

        result = _run(agent.act("Test top_n", verbose=False, return_history=True))
        assert result["success"] is True

        task_dirs = os.listdir(os.path.join(memory_dir, "tasks"))
        trace_path = os.path.join(memory_dir, "tasks", task_dirs[0], "trace.json")
        with open(trace_path) as f:
            trace = json.load(f)["trace"]

        exp_entries = [e for e in trace if any(k.startswith("experience_for_step_") for k in e)]
        assert len(exp_entries) == 1

        key = next(k for k in exp_entries[0] if k.startswith("experience_for_step_"))
        rounds = exp_entries[0][key]
        assert len(rounds) == 1
        assert len(rounds[0]["experiences"]) == top_n, (
            f"Expected {top_n} experiences per round, got {len(rounds[0]['experiences'])}"
        )

        obs = [h for h in result["history"] if h.get("action") == "search_experiences"]
        for i in range(1, top_n + 1):
            assert f"#{i}" in obs[0]["observation"], (
                f"Expected #{i} in observation, got: {obs[0]['observation'][:200]}"
            )

        ok(f"Round returned {top_n} experiences; observation shows #{1}..#{top_n}")
    finally:
        shutil.rmtree(memory_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_search_does_not_count_as_action_turn()
    test_retrieval_logs_interleaved()
    test_reset_state_on_trajectory_change()
    test_max_epoch_enforced_in_loop()
    test_tool_bank_and_schema()
    test_max_total_turns_safety()
    test_flush_before_answer_no_real_tool()
    test_experience_top_n_per_round()
    print("\n🎉 state_level/test_agent_integration ALL PASSED\n")
