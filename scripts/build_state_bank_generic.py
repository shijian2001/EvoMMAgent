#!/usr/bin/env python3
"""Build state-level experience bank using any OpenAI-compatible VL model.

Works with GPT-4o, Gemini, etc. — reuses the same AsyncLLMClient from
error_analysis (httpx + base64), so it has NO model-specific dependencies.

Usage:
    python scripts/build_state_bank_generic.py \
        --memory_dir memory/train_run/ \
        --llm_api_key YOUR_KEY \
        --concurrency 10 \
        --embedding_model Qwen/Qwen3-VL-Embedding-2B \
        --embedding_base_url http://localhost:8001/v1
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
from error_analysis.client import AsyncLLMClient
from mm_memory.state_bank import ALL_VIEWS, StateBank, available_views, compose_view
from mm_memory.retrieval.embedder import Embedder
from scripts.build_state_bank import (
    convert_trace_to_trajectory,
    build_hindsight_prompt,
)

MAX_LONG_SIDE = 768


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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Annotation (uses AsyncLLMClient instead of APIPool)
# ---------------------------------------------------------------------------

SYSTEM_PREFIX = (
    "[System] You are a precise trajectory evaluator. "
    "Always respond in valid JSON.\n\n"
)


async def annotate_single(
    trace_data: Dict[str, Any], client: AsyncLLMClient,
) -> Optional[List[Dict[str, Any]]]:
    """Annotate one trace. Returns annotated trajectory or None."""
    trajectory = convert_trace_to_trajectory(trace_data)
    if not trajectory:
        return None

    prompt, image_paths = build_hindsight_prompt(trace_data, trajectory)
    task_id = trace_data.get("task_id", "?")

    resized_paths, tmp_files = _resize_images(image_paths)
    max_attempts = 8
    try:
        for attempt in range(max_attempts):
            try:
                answer = await client.call(SYSTEM_PREFIX + prompt, resized_paths)
                parsed = JSONParser.parse(answer)

                if isinstance(parsed, list):
                    ann_map = {
                        a.get("state", a.get("state_index")): a
                        for a in parsed if isinstance(a, dict)
                    }
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
                    f"Unexpected parse result for {task_id}: {type(parsed)} "
                    f"(attempt {attempt + 1}/{max_attempts})"
                )
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

    return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def build_state_bank(
    memory_dir: str,
    client: AsyncLLMClient,
    embedder: Embedder,
    batch_size: int = 32,
    concurrency: int = 8,
    trace_mode: str = "both",
    bank_dir_name: str = "state_bank",
) -> None:
    """Full pipeline: scan traces -> annotate -> embed -> save."""
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
    n_correct = n_incorrect = 0
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

    # -- Annotate concurrently --
    sem = asyncio.Semaphore(concurrency)
    annotated: List[Tuple[str, Dict[str, Any], List[Dict[str, Any]]]] = []
    lock = asyncio.Lock()

    async def _annotate(tid: str, td: Dict[str, Any]):
        async with sem:
            traj = await annotate_single(td, client)
            if traj:
                async with lock:
                    annotated.append((tid, td, traj))
                logger.info(
                    f"  Annotated {tid}: {len(traj)} steps, "
                    f"Q-values: {[e.get('q_value', '?') for e in traj]}"
                )

    await asyncio.gather(*[_annotate(tid, td) for tid, td in traces])
    logger.info(f"Annotated {len(annotated)}/{len(traces)} traces")

    if not annotated:
        raise ValueError("No traces were successfully annotated")

    # -- Extract states --
    state_metas: List[Dict[str, Any]] = []
    per_view_indices: Dict[str, List[int]] = {v: [] for v in ALL_VIEWS}
    per_view_payloads: Dict[str, List[Dict[str, Any]]] = {v: [] for v in ALL_VIEWS}

    for task_id, trace_data, trajectory in annotated:
        for entry in trajectory:
            step = entry["state_index"]
            elements = StateBank.state_to_elements(trace_data, trajectory, step)
            avail = set(available_views(elements))
            full_text = compose_view(elements, "all").get("text", "")

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

    # -- Embed and save --
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
        sub_emb = await embedder.encode_view_batch(payloads, batch_size=batch_size)
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
        description="Build state-level experience bank (generic VL model, no qwen_vl_utils)."
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
        "--concurrency", type=int, default=10,
        help="Max concurrent LLM calls (default: 10)",
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
        "--mode", type=str, default="both",
        choices=["correct", "incorrect", "both"],
        help="Which traces to include (default: both)",
    )
    parser.add_argument(
        "--bank_dir_name", type=str, default="state_bank",
        help="Name of the output bank subfolder (default: state_bank)",
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
        max_retries=8,
    )

    embedder = Embedder(
        model_name=args.embedding_model,
        base_url=args.embedding_base_url,
        api_key=args.embedding_api_key,
    )

    async with client:
        await build_state_bank(
            memory_dir=args.memory_dir,
            client=client,
            embedder=embedder,
            batch_size=args.batch_size,
            concurrency=args.concurrency,
            trace_mode=args.mode,
            bank_dir_name=args.bank_dir_name,
        )

    logger.info("Done!")


if __name__ == "__main__":
    asyncio.run(main())
