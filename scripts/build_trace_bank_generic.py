#!/usr/bin/env python3
"""Build trace-level experience bank using any OpenAI-compatible VL model.

Works with GPT-4o, Gemini, etc. — reuses AsyncLLMClient from error_analysis
(httpx + base64), so it has NO model-specific dependencies.

Usage:
    python scripts/build_trace_bank_generic.py \
        --memory_dir memory/train_run/ \
        --llm_api_key YOUR_KEY \
        --concurrency 16 \
        --embedding_model qwen3vl-embed \
        --embedding_base_url http://localhost:8001/v1 \
        --mode both \
        --batch_size 64 \
        --bank_dir_name trace_bank_gpt4o
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

from api.json_parser import JSONParser
from error_analysis.client import AsyncLLMClient, ContentFilterError
from mm_memory.trace_bank import TraceBank
from mm_memory.retrieval.embedder import Embedder
from scripts.build_state_bank import (
    AGENT_TOOLS,
    _build_tools_section,
    _build_trajectory_text,
    convert_trace_to_trajectory,
)

MAX_LONG_SIDE = 768

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _resize_images(image_paths: List[str]) -> Tuple[List[str], List[str]]:
    """Resize images so the long side <= MAX_LONG_SIDE. Returns (resized_paths, tmp_files)."""
    resized = []
    tmp_files = []
    for p in image_paths:
        if not os.path.exists(p):
            continue
        try:
            img = Image.open(p)
            w, h = img.size
            if max(w, h) > MAX_LONG_SIDE:
                scale = MAX_LONG_SIDE / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                img.save(tmp.name, format="JPEG", quality=85)
                tmp.close()
                resized.append(tmp.name)
                tmp_files.append(tmp.name)
            else:
                resized.append(p)
        except Exception:
            resized.append(p)
    return resized, tmp_files


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


# ---------------------------------------------------------------------------
# Hindsight prompt (trace-level: one experience per whole trace)
# ---------------------------------------------------------------------------

CORRECT_PROMPT = """\
You are evaluating a complete agent reasoning trace. The task was solved CORRECTLY.

## Task
{images_note}Question: {question}{type_line}
{tools_section}
## Trajectory
{traj_text}

## Instructions
Write exactly 1-2 concise sentences of actionable experience for future agents
facing similar tasks. Summarize what strategy worked well (tool choice, evidence
usage, reasoning). Keep it generalizable and do not include specific answers.

Output JSON:
{{"experience": "your 1-2 sentence experience"}}"""

INCORRECT_PROMPT = """\
You are evaluating a complete agent reasoning trace. The task was solved INCORRECTLY.

## Task
{images_note}Question: {question}{type_line}
{tools_section}
## Trajectory
{traj_text}

## Instructions
Write exactly 1-2 concise sentences of actionable experience for future agents
facing similar tasks. Identify what went wrong and what should be done differently
(tool choice, evidence usage, common mistakes). Keep it generalizable and do not
include specific answers.

Output JSON:
{{"experience": "your 1-2 sentence experience"}}"""

SYSTEM_PREFIX = (
    "[System] You are a concise trajectory evaluator. "
    "Always respond in valid JSON.\n\n"
)


def _build_hindsight_prompt(
    trace_data: Dict[str, Any],
    trajectory: List[Dict[str, Any]],
) -> Tuple[str, List[str]]:
    question = trace_data.get("input", {}).get("question", "")
    sub_task = trace_data.get("sub_task", "")
    image_paths = _extract_image_paths(trace_data)
    images_note = "The input images are shown above.\n" if image_paths else ""
    type_line = f"\nTask: {sub_task}" if sub_task else ""
    tools_section = _build_tools_section(AGENT_TOOLS)
    traj_text = _build_trajectory_text(trajectory)

    template = CORRECT_PROMPT if trace_data.get("is_correct", False) else INCORRECT_PROMPT
    prompt = template.format(
        images_note=images_note,
        question=question,
        type_line=type_line,
        tools_section=tools_section,
        traj_text=traj_text,
    )
    return prompt, image_paths


# ---------------------------------------------------------------------------
# Annotation (uses AsyncLLMClient)
# ---------------------------------------------------------------------------

async def _annotate_experience(
    trace_data: Dict[str, Any],
    client: AsyncLLMClient,
) -> str:
    """Annotate one trace -> single experience string."""
    trajectory = convert_trace_to_trajectory(trace_data)
    if not trajectory:
        return ""

    prompt, image_paths = _build_hindsight_prompt(trace_data, trajectory)
    task_id = trace_data.get("task_id", "?")

    resized_paths, tmp_files = _resize_images(image_paths)
    max_attempts = 3
    try:
        for attempt in range(max_attempts):
            try:
                answer = await client.call(SYSTEM_PREFIX + prompt, resized_paths)
                parsed = JSONParser.parse(answer)
                if isinstance(parsed, dict):
                    exp = str(parsed.get("experience", "")).strip()
                    if exp:
                        return exp
                logger.warning(
                    f"Unexpected parse for {task_id}: {type(parsed)} "
                    f"(attempt {attempt + 1}/{max_attempts})"
                )
            except ContentFilterError:
                logger.warning(f"Skipping {task_id}: content filter triggered")
                return ""
            except Exception as e:
                logger.warning(
                    f"Annotation failed for {task_id}: {e} "
                    f"(attempt {attempt + 1}/{max_attempts})"
                )
    finally:
        for tmp in tmp_files:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    return ""


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def build_trace_bank(
    memory_dir: str,
    client: AsyncLLMClient,
    embedder: Embedder,
    mode: str = "correct",
    batch_size: int = 32,
    concurrency: int = 8,
    bank_dir_name: str = "trace_bank",
) -> None:
    want_correct = mode in ("correct", "both")
    want_incorrect = mode in ("incorrect", "both")

    tasks_dir = os.path.join(memory_dir, "tasks")
    if not os.path.exists(tasks_dir):
        raise FileNotFoundError(f"Tasks directory not found: {tasks_dir}")

    n_correct = n_incorrect = 0
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

    # -- Annotate concurrently --
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    annotated_pairs: List[Tuple[Dict[str, Any], str]] = []

    async def _worker(td: Dict[str, Any]):
        async with sem:
            exp = await _annotate_experience(td, client)
            if exp:
                async with lock:
                    annotated_pairs.append((td, exp))
                logger.info(f"  Annotated {td.get('task_id', '?')}: {exp[:80]}...")

    await asyncio.gather(*[_worker(t) for t in trace_records])
    logger.info(f"Annotated {len(annotated_pairs)}/{len(trace_records)} traces")

    if not annotated_pairs:
        raise ValueError("No valid hindsight experiences produced.")

    # -- Build embedding payloads --
    payloads: List[Dict[str, Any]] = []
    final_experiences: List[Dict[str, Any]] = []
    for trace_data, exp in annotated_pairs:
        index_text = TraceBank.build_index_text(trace_data)
        if not index_text.strip():
            continue
        payloads.append({
            "text": index_text,
            "images": _extract_image_paths(trace_data),
        })
        final_experiences.append({
            "task_id": trace_data.get("task_id", ""),
            "experience": exp,
            "source": "correct" if trace_data.get("is_correct", False) else "incorrect",
        })

    if not payloads:
        raise ValueError("No valid payloads for embedding.")

    embeddings = await embedder.encode_multimodal_batch(payloads, batch_size=batch_size)
    if embeddings.shape[0] != len(final_experiences):
        raise ValueError("Embeddings and experiences size mismatch.")

    # -- Save --
    bank_dir = os.path.join(memory_dir, bank_dir_name)
    os.makedirs(bank_dir, exist_ok=True)

    np.save(os.path.join(bank_dir, "embeddings.npy"), embeddings)
    with open(os.path.join(bank_dir, "experiences.json"), "w", encoding="utf-8") as f:
        json.dump(final_experiences, f, ensure_ascii=False)

    logger.info(
        f"Trace bank saved to {bank_dir}: "
        f"{len(final_experiences)} entries, dim={embeddings.shape[1]}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(
        description="Build trace-level experience bank (generic VL model, no qwen_vl_utils)."
    )
    parser.add_argument(
        "--memory_dir", type=str, required=True,
        help="Root memory directory containing tasks/*/trace.json",
    )

    # LLM
    parser.add_argument(
        "--llm_api_key", type=str, required=True,
        help="API key for the LLM service",
    )
    parser.add_argument(
        "--concurrency", type=int, default=16,
        help="Max concurrent LLM calls (default: 16)",
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

    # Output
    parser.add_argument(
        "--mode", type=str, default="correct",
        choices=["correct", "incorrect", "both"],
        help="Which traces to include (default: correct)",
    )
    parser.add_argument(
        "--bank_dir_name", type=str, default="trace_bank",
        help="Name of the output bank subfolder (default: trace_bank)",
    )

    args = parser.parse_args()

    logger.info(f"Memory dir:  {args.memory_dir}")
    logger.info(f"Concurrency: {args.concurrency}")
    logger.info(f"Embedding:   {args.embedding_model}")
    logger.info(f"Trace mode:  {args.mode}")
    logger.info(f"Bank dir:    {args.bank_dir_name}")

    client = AsyncLLMClient(
        api_key=args.llm_api_key,
        concurrency=args.concurrency,
        max_retries=3,
    )

    embedder = Embedder(
        model_name=args.embedding_model,
        base_url=args.embedding_base_url,
        api_key=args.embedding_api_key,
    )

    async with client:
        await build_trace_bank(
            memory_dir=args.memory_dir,
            client=client,
            embedder=embedder,
            mode=args.mode,
            batch_size=args.batch_size,
            concurrency=args.concurrency,
            bank_dir_name=args.bank_dir_name,
        )

    logger.info("Done!")


if __name__ == "__main__":
    asyncio.run(main())
