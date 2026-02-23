#!/usr/bin/env python3
"""Offline script to build a state-level experience bank.

Scans correct traces, converts to MDP trajectories, uses LLM hindsight to
annotate Q-values and experiences, embeds states, and saves the state bank.
All processing happens in memory — original trace.json files are NEVER modified.

Usage:
    python scripts/build_state_bank.py \
        --memory_dir memory/train_run/ \
        --llm_model qwen3-vl-32b-instruct \
        --llm_base_url https://maas.devops.xiaohongshu.com/v1 \
        --llm_api_key YOUR_KEY \
        --embedding_model Qwen/Qwen3-VL-Embedding-2B \
        --embedding_base_url http://localhost:8001/v1
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path so imports work when running as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from api.json_parser import JSONParser
from mm_memory.state_bank import StateBank
from mm_memory.retrieval.embedder import Embedder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trace → MDP trajectory conversion
# ---------------------------------------------------------------------------

def convert_trace_to_trajectory(trace_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert raw trace steps (think/action/answer) into MDP trajectory.

    Merges adjacent think + action/answer into atomic action entries:
        a_t = (thinking, tool, parameters, observation)
    """
    raw_steps = trace_data.get("trace", [])
    trajectory: List[Dict[str, Any]] = []
    current_thinking = ""

    for step in raw_steps:
        stype = step.get("type", "")

        if stype == "think":
            current_thinking = step.get("content", "")

        elif stype == "action":
            obs = step.get("observation", "")
            if isinstance(obs, dict):
                obs = obs.get("description", str(obs))
            trajectory.append({
                "state_index": len(trajectory),
                "action": {
                    "thinking": current_thinking,
                    "tool": step.get("tool", ""),
                    "parameters": step.get("properties", {}),
                    "observation": str(obs),
                },
            })
            current_thinking = ""

        elif stype == "answer":
            # answer is a terminal decision — include as an action entry
            trajectory.append({
                "state_index": len(trajectory),
                "action": {
                    "thinking": current_thinking,
                    "tool": "answer",
                    "parameters": {"content": step.get("content", "")},
                    "observation": None,
                },
            })
            current_thinking = ""

    return trajectory


# ---------------------------------------------------------------------------
# Hindsight annotation (LLM)
# ---------------------------------------------------------------------------

def build_hindsight_prompt(
    trace_data: Dict[str, Any], trajectory: List[Dict[str, Any]]
) -> Tuple[str, List[str]]:
    """Build prompt for LLM to annotate Q-values and experiences.

    Returns:
        Tuple of (prompt_text, image_paths) where image_paths may be empty.
    """
    question = trace_data.get("input", {}).get("question", "")
    answer = trace_data.get("answer", "")
    sub_task = trace_data.get("sub_task", "")
    # Collect input image paths (if any)
    image_paths = [
        img["path"] for img in trace_data.get("input", {}).get("images", [])
        if isinstance(img, dict) and img.get("path")
    ]

    traj_lines = [
        "State s_0:\n  - The question and input images shown above."
    ]
    for entry in trajectory:
        idx = entry["state_index"]
        a = entry["action"]
        tool = a["tool"]
        params = a.get("parameters", {})
        obs = a.get("observation") or ""
        think = a.get("thinking", "")

        action_lines = [f"Action a_{idx}:"]
        if think:
            action_lines.append(f"  Think: {think}")

        if tool == "answer":
            action_lines.append("  Tool: answer")
            action_lines.append(f"  Answer content: {params.get('content', '')}")
        else:
            action_lines.append(f"  Tool: {tool}({params})")
            if obs:
                action_lines.append(f"  Observation: {obs}")

        traj_lines.append("\n".join(action_lines))

        if idx + 1 < len(trajectory):
            traj_lines.append(f"State s_{idx + 1}")

    traj_text = "\n\n".join(traj_lines)

    images_note = (
        "The input images are shown above.\n"
        if image_paths else ""
    )
    type_line = f"\nType: {sub_task}" if sub_task else ""

    prompt = f"""\
You are evaluating a complete agent reasoning trace. The task was solved CORRECTLY.
The process may be highly efficient, or it may contain wasteful, redundant, or
error-prone steps. Judge each step by the quality and relevance of its output.

## Task
{images_note}Question: {question}{type_line}

## Trajectory
The initial state s_0 is the question and images above. Each subsequent state
is what the agent has seen so far, including all previous actions and observations.

{traj_text}

## Instructions
For each (state, action) pair, provide:

1. q_value (0-10): Rate the quality of this action at this state.
   A correct final answer does NOT mean every step was good. Judge each step
   by whether it produced accurate, relevant information that actually
   contributed to reaching the answer.
   - 9-10: Essential — produced decisive information that directly shaped
           the answer, or answered correctly at the right time
   - 7-8: Helpful — useful output that advanced the solution, with minor
           room for improvement in tool choice or parameters
   - 5-6: Reasonable — valid approach, but the output had little actual
           influence on the final answer; a more direct path existed
   - 3-4: Wasteful — produced no new information, duplicated prior knowledge,
           or was an unnecessary detour
   - 0-2: Harmful — produced errors, misleading output, or repeated a
           known failure

2. experience (1-2 sentences): Actionable advice that a FUTURE agent would see
   BEFORE making this decision. This advice will be injected into the agent's
   context at runtime to guide its next action.
   Guidelines:
   - Reference the task type and tool strategy, but keep the advice
     generalizable — do not mention specific objects, labels, or values
     from this particular trace
   - Focus on STRATEGY: recommend the specific tool and approach that works
     for this task type, or advise answering directly if tools are unnecessary
   - If the step was an error, redundant, or unnecessary, the experience
     MUST be cautionary — warn against this approach rather than endorsing it
   - NEVER mention or hint at the correct answer
   Good: "For visual similarity tasks, use get_image2images_similarity to get
          objective scores rather than estimating visually."
   Good: "The observation already contains enough information to answer directly.
          No further tool calls are needed."
   Good: "Avoid calling additional tools — the current observations already
          provide sufficient evidence to answer directly."
   Bad:  "The agent should use the right tool." (too vague)
   Bad:  "The answer is likely A based on the scores." (leaks answer)

Output JSON array:
[
  {{"state": 0, "q_value": 8, "experience": "..."}},
  ...
]"""
    return prompt, image_paths


async def annotate_single(
    trace_data: Dict[str, Any], api_pool
) -> Optional[List[Dict[str, Any]]]:
    """Annotate one trace in memory. Returns annotated trajectory or None."""
    trajectory = convert_trace_to_trajectory(trace_data)
    if not trajectory:
        return None

    prompt, image_paths = build_hindsight_prompt(trace_data, trajectory)

    # Build user message: images first, then text prompt
    if image_paths:
        user_content: Any = []
        for img_path in image_paths:
            if os.path.exists(img_path):
                user_content.append({"type": "image", "image": img_path})
        user_content.append({"type": "text", "text": prompt})
    else:
        user_content = prompt

    try:
        result = await api_pool.execute(
            "qa",
            system="You are a precise trajectory evaluator. Always respond in valid JSON.",
            messages=[{"role": "user", "content": user_content}],
            temperature=0.2,
            max_tokens=2000,
        )
        answer = result.get("answer", "")
        parsed = JSONParser.parse(answer)

        if isinstance(parsed, list):
            ann_map = {a.get("state", a.get("state_index")): a for a in parsed if isinstance(a, dict)}
            for entry in trajectory:
                idx = entry["state_index"]
                if idx in ann_map:
                    entry["q_value"] = ann_map[idx].get("q_value", 5)
                    entry["experience"] = ann_map[idx].get("experience", "")
                else:
                    entry["q_value"] = 5
                    entry["experience"] = ""
            return trajectory

        logger.warning(f"Unexpected parse result type: {type(parsed)}")
        return None

    except Exception as e:
        task_id = trace_data.get("task_id", "?")
        logger.warning(f"Annotation failed for {task_id}: {e}")
        return None


# ---------------------------------------------------------------------------
# Main pipeline: annotate → extract states → embed → save
# ---------------------------------------------------------------------------

async def build_state_bank(
    memory_dir: str,
    api_pool,
    embedder: Embedder,
    min_q: int = 5,
    batch_size: int = 32,
    concurrency: int = 8,
) -> None:
    """Full pipeline: scan traces → annotate → embed → save state_bank/.

    Original trace.json files are never modified.
    """
    tasks_dir = os.path.join(memory_dir, "tasks")
    if not os.path.exists(tasks_dir):
        raise FileNotFoundError(f"Tasks directory not found: {tasks_dir}")

    # Scan correct traces
    task_dirs = sorted(
        d for d in os.listdir(tasks_dir)
        if os.path.isdir(os.path.join(tasks_dir, d))
    )

    traces: List[Tuple[str, Dict[str, Any]]] = []  # (task_id, trace_data)
    for task_id in task_dirs:
        trace_path = os.path.join(tasks_dir, task_id, "trace.json")
        if not os.path.exists(trace_path):
            continue
        try:
            with open(trace_path, "r", encoding="utf-8") as f:
                trace = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if not trace.get("is_correct", False):
            continue
        traces.append((task_id, trace))

    logger.info(f"Found {len(traces)} correct traces")

    # ── Annotate concurrently (in memory) ──
    sem = asyncio.Semaphore(concurrency)
    # Collect (task_id, trace_data, annotated_trajectory) tuples
    annotated: List[Tuple[str, Dict[str, Any], List[Dict[str, Any]]]] = []
    lock = asyncio.Lock()

    async def _annotate(task_id: str, trace_data: Dict[str, Any]):
        async with sem:
            trajectory = await annotate_single(trace_data, api_pool)
            if trajectory:
                async with lock:
                    annotated.append((task_id, trace_data, trajectory))
                logger.info(
                    f"  Annotated {task_id}: {len(trajectory)} steps, "
                    f"Q-values: {[e.get('q_value', '?') for e in trajectory]}"
                )

    await asyncio.gather(*[_annotate(tid, td) for tid, td in traces])
    logger.info(f"Annotated {len(annotated)}/{len(traces)} traces")

    if not annotated:
        raise ValueError("No traces were successfully annotated")

    # ── Extract states with Q >= min_q ──
    state_texts: List[str] = []
    state_metas: List[Dict[str, Any]] = []

    for task_id, trace_data, trajectory in annotated:
        for entry in trajectory:
            q = entry.get("q_value", 0)
            if q < min_q:
                continue
            step = entry["state_index"]
            text = StateBank.state_to_text(trace_data, trajectory, step)
            if not text.strip():
                continue
            state_texts.append(text)
            state_metas.append({
                "task_id": task_id,
                "state": step,
                "experience": entry.get("experience", ""),
                "q_value": q,
                "state_text": text,
            })

    if not state_texts:
        raise ValueError(f"No states with Q >= {min_q} found")

    logger.info(f"Extracted {len(state_texts)} states (Q >= {min_q})")

    # ── Embed and save ──
    embeddings = await embedder.encode_batch(state_texts, batch_size=batch_size)

    bank_dir = os.path.join(memory_dir, "state_bank")
    os.makedirs(bank_dir, exist_ok=True)

    np.save(os.path.join(bank_dir, "embeddings.npy"), embeddings)
    with open(os.path.join(bank_dir, "state_meta.json"), "w", encoding="utf-8") as f:
        json.dump(state_metas, f, ensure_ascii=False, indent=2)

    logger.info(
        f"State bank saved to {bank_dir}: "
        f"{len(state_metas)} states, dim={embeddings.shape[1]}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(
        description="Build a state-level experience bank (annotate + embed)."
    )
    parser.add_argument(
        "--memory_dir", type=str, required=True,
        help="Root memory directory containing tasks/*/trace.json",
    )

    # LLM for hindsight annotation
    parser.add_argument(
        "--llm_model", type=str, required=True,
        help="LLM model for hindsight annotation",
    )
    parser.add_argument(
        "--llm_base_url", type=str, required=True,
        help="LLM API base URL",
    )
    parser.add_argument(
        "--llm_api_key", type=str, default="",
        help="API key for the LLM service",
    )
    parser.add_argument(
        "--concurrency", type=int, default=8,
        help="Max concurrent LLM calls for annotation (default: 8)",
    )

    # Embedding
    parser.add_argument(
        "--embedding_model", type=str, required=True,
        help="Name of the embedding model deployed on vLLM",
    )
    parser.add_argument(
        "--embedding_base_url", type=str, required=True,
        help="vLLM embedding endpoint (e.g. http://host:8001/v1)",
    )
    parser.add_argument(
        "--embedding_api_key", type=str, default="dummy",
        help="API key for the embedding service (default: dummy)",
    )
    parser.add_argument(
        "--min_q", type=int, default=7,
        help="Minimum Q-value to include a state (default: 7)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for embedding API calls (default: 32)",
    )

    args = parser.parse_args()

    logger.info(f"Memory dir: {args.memory_dir}")
    logger.info(f"LLM: {args.llm_model}")
    logger.info(f"Embedding: {args.embedding_model}")
    logger.info(f"Min Q-value: {args.min_q}")

    from api.async_pool import APIPool
    api_pool = APIPool(
        model_name=args.llm_model,
        api_keys=[args.llm_api_key] if args.llm_api_key else ["dummy"],
        base_url=args.llm_base_url,
        max_concurrent_per_key=5,
    )

    embedder = Embedder(
        model_name=args.embedding_model,
        base_url=args.embedding_base_url,
        api_key=args.embedding_api_key,
    )

    await build_state_bank(
        memory_dir=args.memory_dir,
        api_pool=api_pool,
        embedder=embedder,
        min_q=args.min_q,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
    )

    logger.info("Done!")


if __name__ == "__main__":
    asyncio.run(main())
