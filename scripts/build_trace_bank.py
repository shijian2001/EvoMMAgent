#!/usr/bin/env python3
"""Build trace bank with precomputed hindsight experiences.

Pipeline:
1) Scan tasks/*/trace.json and filter traces
2) Generate one 1-2 sentence hindsight experience per trace via VLM
3) Build multimodal index embeddings from (images + Question + Task)
4) Save trace_bank/{embeddings.npy, experiences.json}
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root to path so imports work when running as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.async_pool import APIPool
from api.json_parser import JSONParser
from mm_memory.retrieval.embedder import Embedder
from mm_memory.trace_bank import TraceBank
from scripts.build_state_bank import (
    AGENT_TOOLS,
    _build_tools_section,
    _build_trajectory_text,
    convert_trace_to_trajectory,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


HINDSIGHT_PROMPT = """\
You are evaluating a complete agent reasoning trace.

## Task
{images_note}Question: {question}{type_line}
{tools_section}
## Trajectory
{traj_text}

## Instructions
Write exactly 1-2 concise sentences of actionable experience for future agents
facing similar tasks. Focus on strategy (tool choice, evidence usage, and common
mistakes). Keep it generalizable and do not include specific answers.

Output JSON:
{{"experience": "your 1-2 sentence experience"}}"""


def _extract_image_paths(trace_data: Dict[str, Any]) -> List[str]:
    image_paths: List[str] = []
    for item in trace_data.get("input", {}).get("images", []):
        if isinstance(item, dict):
            path = item.get("path")
        else:
            path = item
        if path:
            image_paths.append(path)
    return image_paths


def _build_hindsight_prompt(trace_data: Dict[str, Any], trajectory: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    question = trace_data.get("input", {}).get("question", "")
    sub_task = trace_data.get("sub_task", "")
    image_paths = _extract_image_paths(trace_data)
    images_note = "The input images are shown above.\n" if image_paths else ""
    type_line = f"\nTask: {sub_task}" if sub_task else ""
    tools_section = _build_tools_section(AGENT_TOOLS)
    traj_text = _build_trajectory_text(trajectory)

    prompt = HINDSIGHT_PROMPT.format(
        images_note=images_note,
        question=question,
        type_line=type_line,
        tools_section=tools_section,
        traj_text=traj_text,
    )
    return prompt, image_paths


async def _annotate_experience(trace_data: Dict[str, Any], api_pool: APIPool) -> str:
    trajectory = convert_trace_to_trajectory(trace_data)
    if not trajectory:
        return ""

    prompt, image_paths = _build_hindsight_prompt(trace_data, trajectory)
    if image_paths:
        user_content: Any = []
        for img_path in image_paths:
            if os.path.exists(img_path):
                user_content.append({"type": "image", "image": img_path})
        user_content.append({"type": "text", "text": prompt})
    else:
        user_content = prompt

    for _ in range(3):
        try:
            result = await api_pool.execute(
                "qa",
                system="You are a concise trajectory evaluator. Always output valid JSON.",
                messages=[{"role": "user", "content": user_content}],
                temperature=0.2,
                max_tokens=300,
            )
            parsed = JSONParser.parse(result.get("answer", ""))
            if isinstance(parsed, dict):
                exp = str(parsed.get("experience", "")).strip()
                if exp:
                    return exp
        except Exception as e:
            logger.warning(f"Hindsight annotation failed: {e}")

    return ""


async def build_trace_bank(
    memory_dir: str,
    api_pool: APIPool,
    embedder: Embedder,
    mode: str = "correct",
    batch_size: int = 32,
    hindsight_concurrency: int = 8,
    bank_dir_name: str = "trace_bank",
) -> None:
    want_correct = mode in ("correct", "both")
    want_incorrect = mode in ("incorrect", "both")

    tasks_dir = os.path.join(memory_dir, "tasks")
    if not os.path.exists(tasks_dir):
        raise FileNotFoundError(f"Tasks directory not found: {tasks_dir}")

    n_correct = 0
    n_incorrect = 0
    trace_records: List[Dict[str, Any]] = []
    for task_id in sorted(os.listdir(tasks_dir)):
        trace_path = os.path.join(tasks_dir, task_id, "trace.json")
        if not os.path.exists(trace_path):
            continue
        try:
            with open(trace_path, "r", encoding="utf-8") as f:
                trace_data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Skipping {task_id}: {e}")
            continue

        is_correct = bool(trace_data.get("is_correct", False))
        if is_correct and want_correct:
            n_correct += 1
            trace_data["task_id"] = task_id
            trace_records.append(trace_data)
        elif not is_correct and want_incorrect:
            n_incorrect += 1
            trace_data["task_id"] = task_id
            trace_records.append(trace_data)

    logger.info(
        f"Found {n_correct} correct + {n_incorrect} incorrect = {len(trace_records)} traces "
        f"(mode={mode})"
    )

    if not trace_records:
        raise ValueError("No traces found after filtering.")

    sem = asyncio.Semaphore(hindsight_concurrency)

    async def _worker(trace_data: Dict[str, Any]) -> str:
        async with sem:
            return await _annotate_experience(trace_data, api_pool)

    experiences = await asyncio.gather(*[_worker(t) for t in trace_records])

    payloads: List[Dict[str, Any]] = []
    final_experiences: List[str] = []
    for trace_data, exp in zip(trace_records, experiences):
        if not exp:
            continue
        index_text = TraceBank.build_index_text(trace_data)
        if not index_text.strip():
            continue
        payloads.append(
            {
                "text": index_text,
                "images": _extract_image_paths(trace_data),
            }
        )
        final_experiences.append(exp)

    if not payloads:
        raise ValueError("No valid hindsight experiences produced.")

    embeddings = await embedder.encode_multimodal_batch(payloads, batch_size=batch_size)
    if embeddings.shape[0] != len(final_experiences):
        raise ValueError("Embeddings and experiences size mismatch.")

    bank_dir = os.path.join(memory_dir, bank_dir_name)
    os.makedirs(bank_dir, exist_ok=True)

    np.save(os.path.join(bank_dir, "embeddings.npy"), embeddings)
    with open(os.path.join(bank_dir, "experiences.json"), "w", encoding="utf-8") as f:
        json.dump(final_experiences, f, ensure_ascii=False)

    logger.info(
        f"Trace bank saved to {bank_dir}: "
        f"{len(final_experiences)} entries, dim={embeddings.shape[1]}"
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Build trace-level hindsight bank.")
    parser.add_argument("--memory_dir", type=str, required=True)
    parser.add_argument("--llm_model", type=str, required=True)
    parser.add_argument("--llm_base_url", type=str, required=True)
    parser.add_argument("--llm_api_key", type=str, default="")
    parser.add_argument("--embedding_model", type=str, required=True)
    parser.add_argument("--embedding_base_url", type=str, required=True)
    parser.add_argument("--embedding_api_key", type=str, default="dummy")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hindsight_concurrency", type=int, default=8)
    parser.add_argument(
        "--mode", type=str, default="correct",
        choices=["correct", "incorrect", "both"],
        help="Which traces to include (default: correct)",
    )
    parser.add_argument("--bank_dir_name", type=str, default="trace_bank")
    args = parser.parse_args()

    api_pool = APIPool(
        model_name=args.llm_model,
        api_keys=[args.llm_api_key or "dummy"],
        base_url=args.llm_base_url,
        max_concurrent_per_key=max(1, args.hindsight_concurrency),
    )
    embedder = Embedder(
        model_name=args.embedding_model,
        base_url=args.embedding_base_url,
        api_key=args.embedding_api_key,
    )

    await build_trace_bank(
        memory_dir=args.memory_dir,
        api_pool=api_pool,
        embedder=embedder,
        mode=args.mode,
        batch_size=args.batch_size,
        hindsight_concurrency=args.hindsight_concurrency,
        bank_dir_name=args.bank_dir_name,
    )


if __name__ == "__main__":
    asyncio.run(main())
