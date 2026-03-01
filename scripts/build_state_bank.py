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
from mm_memory.state_bank import ALL_VIEWS, StateBank, available_views, compose_view
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

def _build_trajectory_text(trajectory: List[Dict[str, Any]]) -> str:
    """Render trajectory entries into human-readable text."""
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

    return "\n\n".join(traj_lines)


def _build_correct_prompt(
    question: str, images_note: str, type_line: str, traj_text: str,
    tools_section: str = "",
) -> str:
    return f"""\
You are evaluating a complete agent reasoning trace. The task was solved CORRECTLY.
The process may be highly efficient, or it may contain wasteful, redundant, or
error-prone steps. Judge each step by the quality and relevance of its output.

## Task
{images_note}Question: {question}{type_line}
{tools_section}
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


def _build_incorrect_prompt(
    question: str, images_note: str, type_line: str, traj_text: str,
    ground_truth: str, tools_section: str = "",
) -> str:
    gt_line = f"\nCorrect answer: {ground_truth}" if ground_truth else ""
    return f"""\
You are evaluating a complete agent reasoning trace. The task was solved INCORRECTLY.{gt_line}

Your goal is to identify which steps were most responsible for the incorrect outcome.
A wrong final answer does NOT mean every step was bad — many steps may have been
perfectly reasonable. Focus on pinpointing the critical steps that introduced errors,
ignored evidence, or led the reasoning astray.

## Task
{images_note}Question: {question}{type_line}
{tools_section}
## Trajectory
The initial state s_0 is the question and images above. Each subsequent state
is what the agent has seen so far, including all previous actions and observations.

{traj_text}

## Instructions
For each (state, action) pair, provide:

1. q_value (0-10): Rate how much this step CONTRIBUTED TO the incorrect outcome.
   A high score means this step was a critical cause of the error — it introduced
   wrong information, ignored available evidence, or misled subsequent reasoning.
   A low score means this step was irrelevant to the failure (neutral or even
   reasonable on its own).
   - 9-10: Critical error — this step directly caused or cemented the wrong
           answer (e.g. misread evidence, wrong tool, flawed conclusion)
   - 7-8: Significant — substantially misled the reasoning, even if not the
           sole cause of failure
   - 5-6: Moderate — somewhat contributed to the error, but other steps were
           more decisive
   - 3-4: Minor — had little impact on the incorrect outcome; the step was
           mostly neutral
   - 0-2: Irrelevant — this step was reasonable and did not contribute to
           the failure at all

2. experience (1-2 sentences): Actionable advice that a FUTURE agent would see
   BEFORE making this decision. This advice will be injected into the agent's
   context at runtime to guide its next action.
   Guidelines:
   - For high q_value steps (critical errors): the experience MUST be a
     cautionary warning — explain what went wrong and what to avoid
   - For low q_value steps (irrelevant to failure): provide brief neutral or
     positive guidance if appropriate
   - Reference the task type and tool strategy, but keep the advice
     generalizable — do not mention specific objects, labels, or values
     from this particular trace
   - NEVER mention or hint at the correct answer
   Good: "For this task type, do not rely on visual estimation alone — use
          get_image2images_similarity for objective comparison."
   Good: "When observations conflict with prior reasoning, re-examine the
          evidence before committing to an answer."
   Good: "Avoid repeating the same tool call when it has already returned
          a clear result — analyze existing observations instead."
   Bad:  "The agent should be more careful." (too vague)
   Bad:  "The answer is likely A based on the scores." (leaks answer)

Output JSON array:
[
  {{"state": 0, "q_value": 8, "experience": "..."}},
  ...
]"""


AGENT_TOOLS = [
    ("ocr", "Extract text from an image"),
    ("solve_math_equation", "Solve mathematical equations and problems using WolframAlpha"),
    ("web_search", "Search the web for information using a text query"),
    ("localize_objects", "Localize objects and return their bounding boxes"),
    ("zoom_in", "Zoom in on a region by cropping and upscaling"),
    ("calculator", "Calculate mathematical expressions"),
    ("crop", "Crop a region from an image"),
    ("visualize_regions", "Draw bounding boxes and labels on an image"),
    ("estimate_region_depth", "Estimate depth of a region"),
    ("estimate_object_depth", "Estimate depth of an object by text description"),
    ("get_image2images_similarity", "Compute similarity between one image and multiple images"),
    ("get_image2texts_similarity", "Compute similarity between one image and multiple texts"),
    ("get_text2images_similarity", "Compute similarity between one text and multiple images"),
]


def _build_tools_section(tools: List[Tuple[str, str]]) -> str:
    """Build an '## Available Tools' section for hindsight prompts."""
    if not tools:
        return ""
    lines = ["## Available Tools"]
    for name, desc in tools:
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines) + "\n"


def build_hindsight_prompt(
    trace_data: Dict[str, Any], trajectory: List[Dict[str, Any]]
) -> Tuple[str, List[str]]:
    """Build prompt for LLM to annotate Q-values and experiences.

    Selects correct/incorrect prompt variant based on ``trace_data["is_correct"]``.

    Returns:
        Tuple of (prompt_text, image_paths) where image_paths may be empty.
    """
    question = trace_data.get("input", {}).get("question", "")
    sub_task = trace_data.get("sub_task", "")
    image_paths = [
        img["path"] for img in trace_data.get("input", {}).get("images", [])
        if isinstance(img, dict) and img.get("path")
    ]

    traj_text = _build_trajectory_text(trajectory)
    images_note = "The input images are shown above.\n" if image_paths else ""
    type_line = f"\nType: {sub_task}" if sub_task else ""
    tools_section = _build_tools_section(AGENT_TOOLS)

    if trace_data.get("is_correct", False):
        prompt = _build_correct_prompt(
            question, images_note, type_line, traj_text, tools_section,
        )
    else:
        ground_truth = trace_data.get("ground_truth", "")
        prompt = _build_incorrect_prompt(
            question, images_note, type_line, traj_text, ground_truth,
            tools_section,
        )

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

    task_id = trace_data.get("task_id", "?")

    for attempt in range(3):
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

            logger.warning(
                f"Unexpected parse result type for {task_id}: {type(parsed)} "
                f"(attempt {attempt + 1}/3)"
            )

        except Exception as e:
            logger.warning(f"Annotation failed for {task_id}: {e} (attempt {attempt + 1}/3)")

    return None


# ---------------------------------------------------------------------------
# Main pipeline: annotate → extract states → embed → save
# ---------------------------------------------------------------------------

async def build_state_bank(
    memory_dir: str,
    api_pool,
    embedder: Embedder,
    batch_size: int = 32,
    concurrency: int = 8,
    trace_mode: str = "both",
    bank_dir_name: str = "state_bank",
) -> None:
    """Full pipeline: scan traces → annotate → embed → save ``{bank_dir_name}/``.

    Original trace.json files are never modified.

    Args:
        trace_mode: Which traces to include — "correct", "incorrect", or "both".
            "incorrect" requires ``ground_truth`` field in trace.json.
        bank_dir_name: Name of the output subfolder under *memory_dir*.
    """
    tasks_dir = os.path.join(memory_dir, "tasks")
    if not os.path.exists(tasks_dir):
        raise FileNotFoundError(f"Tasks directory not found: {tasks_dir}")

    task_dirs = sorted(
        d for d in os.listdir(tasks_dir)
        if os.path.isdir(os.path.join(tasks_dir, d))
    )

    want_correct = trace_mode in ("correct", "both")
    want_incorrect = trace_mode in ("incorrect", "both")

    traces: List[Tuple[str, Dict[str, Any]]] = []
    n_correct = 0
    n_incorrect = 0
    for task_id in task_dirs:
        trace_path = os.path.join(tasks_dir, task_id, "trace.json")
        if not os.path.exists(trace_path):
            continue
        try:
            with open(trace_path, "r", encoding="utf-8") as f:
                trace = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        is_correct = trace.get("is_correct", False)
        if is_correct and want_correct:
            n_correct += 1
            traces.append((task_id, trace))
        elif not is_correct and want_incorrect:
            n_incorrect += 1
            traces.append((task_id, trace))

    logger.info(
        f"Found {n_correct} correct + {n_incorrect} incorrect = {len(traces)} traces "
        f"(mode={trace_mode})"
    )

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

    # ── Extract all annotated states (Q-value filtering deferred to retrieval) ──
    state_metas: List[Dict[str, Any]] = []
    per_view_indices: Dict[str, List[int]] = {view: [] for view in ALL_VIEWS}
    per_view_payloads: Dict[str, List[Dict[str, Any]]] = {view: [] for view in ALL_VIEWS}

    for task_id, trace_data, trajectory in annotated:
        for entry in trajectory:
            step = entry["state_index"]
            elements = StateBank.state_to_elements(trace_data, trajectory, step)
            avail = set(available_views(elements))

            # Keep a full text snapshot for debugging/readability.
            full_text = compose_view(elements, "question+task+images+observations").get("text", "")

            meta_idx = len(state_metas)
            state_metas.append({
                "task_id": task_id,
                "state": step,
                "experience": entry.get("experience", ""),
                "q_value": entry.get("q_value", 0),
                "state_text": full_text,
                "elements": elements,
                "available_views": sorted(list(avail)),
                "source": "correct" if trace_data.get("is_correct", True) else "incorrect",
            })

            for view in ALL_VIEWS:
                if view not in avail:
                    continue
                per_view_indices[view].append(meta_idx)
                per_view_payloads[view].append(compose_view(elements, view))

    if not state_metas:
        raise ValueError("No states found after annotation")

    logger.info(f"Extracted {len(state_metas)} states")

    # ── Embed and save multi-view bank ──
    bank_dir = os.path.join(memory_dir, bank_dir_name)
    views_dir = os.path.join(bank_dir, "views")
    os.makedirs(views_dir, exist_ok=True)

    global_dim: Optional[int] = None
    for view in ALL_VIEWS:
        indices = per_view_indices[view]
        payloads = per_view_payloads[view]
        if not indices:
            continue

        logger.info(f"Embedding view {view}: {len(payloads)} states")
        sub_emb = await embedder.encode_multimodal_batch(payloads, batch_size=batch_size)
        if global_dim is None:
            global_dim = sub_emb.shape[1]

        full_emb = np.zeros((len(state_metas), sub_emb.shape[1]), dtype=np.float32)
        full_mask = np.zeros((len(state_metas),), dtype=bool)
        for row_i, state_i in enumerate(indices):
            full_emb[state_i] = sub_emb[row_i]
            full_mask[state_i] = True

        np.save(os.path.join(views_dir, f"{view}.npy"), full_emb)
        np.save(os.path.join(views_dir, f"{view}_mask.npy"), full_mask)

    with open(os.path.join(bank_dir, "state_meta.json"), "w", encoding="utf-8") as f:
        json.dump(state_metas, f, ensure_ascii=False, indent=2)

    logger.info(
        f"State bank saved to {bank_dir}: "
        f"{len(state_metas)} states, dim={global_dim if global_dim is not None else 'unknown'}"
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
        "--batch_size", type=int, default=32,
        help="Batch size for embedding API calls (default: 32)",
    )
    parser.add_argument(
        "--mode", type=str, default="both",
        choices=["correct", "incorrect", "both"],
        help="Which traces to include (default: both)",
    )
    parser.add_argument(
        "--bank_dir_name", type=str, default="state_bank",
        help="Name of the output bank subfolder (default: state_bank)",
    )

    args = parser.parse_args()

    logger.info(f"Memory dir: {args.memory_dir}")
    logger.info(f"LLM: {args.llm_model}")
    logger.info(f"Embedding: {args.embedding_model}")
    logger.info(f"Trace mode: {args.mode}")
    logger.info(f"Bank dir name: {args.bank_dir_name}")

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
        batch_size=args.batch_size,
        concurrency=args.concurrency,
        trace_mode=args.mode,
        bank_dir_name=args.bank_dir_name,
    )

    logger.info("Done!")


if __name__ == "__main__":
    asyncio.run(main())
