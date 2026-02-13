#!/usr/bin/env python3
"""Offline script to build a memory bank from training traces.

Scans {memory_dir}/tasks/*/trace.json, filters correct traces,
optionally generates image captions via a VLM, computes embeddings,
and saves the bank to {memory_dir}/trace_bank/.

Usage:
    # Without captions (text-only index)
    python scripts/build_memory_bank.py \
        --memory_dir memory/train_run/ \
        --embedding_model Qwen/Qwen3-VL-Embedding-2B \
        --embedding_base_url http://localhost:8001/v1

    # With captions (multimodal index — requires a VLM)
    python scripts/build_memory_bank.py \
        --memory_dir memory/train_run/ \
        --embedding_model Qwen/Qwen3-VL-Embedding-2B \
        --embedding_base_url http://localhost:8001/v1 \
        --llm_model qwen3-vl-32b-instruct \
        --llm_base_url https://maas.devops.xiaohongshu.com/v1 \
        --llm_api_key YOUR_KEY
"""

import argparse
import asyncio
import logging
import sys
import os

# Add project root to path so imports work when running as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mm_memory.memory_bank import MemoryBank
from mm_memory.retrieval.embedder import Embedder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(
        description="Build a memory bank from training traces."
    )
    parser.add_argument(
        "--memory_dir",
        type=str,
        required=True,
        help="Root memory directory containing tasks/*/trace.json",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        required=True,
        help="Name of the embedding model deployed on vLLM",
    )
    parser.add_argument(
        "--embedding_base_url",
        type=str,
        required=True,
        help="vLLM embedding endpoint (e.g. http://host:8001/v1)",
    )
    parser.add_argument(
        "--embedding_api_key",
        type=str,
        default="dummy",
        help="API key for the embedding service (default: dummy)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for embedding API calls (default: 32)",
    )
    parser.add_argument(
        "--no_filter_correct",
        action="store_true",
        help="Include all traces, not just correct ones",
    )
    # Caption generation (optional — requires a multimodal LLM)
    parser.add_argument(
        "--llm_model",
        type=str,
        default="",
        help="VLM model name for caption generation (omit to skip captions)",
    )
    parser.add_argument(
        "--llm_base_url",
        type=str,
        default="",
        help="VLM API base URL (e.g. https://maas.devops.xiaohongshu.com/v1)",
    )
    parser.add_argument(
        "--llm_api_key",
        type=str,
        default="",
        help="API key for the VLM service",
    )

    args = parser.parse_args()

    logger.info(f"Building memory bank from: {args.memory_dir}")
    logger.info(f"Embedding model: {args.embedding_model}")
    logger.info(f"Embedding endpoint: {args.embedding_base_url}")
    logger.info(f"Filter correct only: {not args.no_filter_correct}")
    logger.info(f"Batch size: {args.batch_size}")

    embedder = Embedder(
        model_name=args.embedding_model,
        base_url=args.embedding_base_url,
        api_key=args.embedding_api_key,
    )

    # Create APIPool for caption generation if LLM args provided
    api_pool = None
    if args.llm_model and args.llm_base_url:
        from api.async_pool import APIPool

        api_keys = [args.llm_api_key] if args.llm_api_key else ["dummy"]
        api_pool = APIPool(
            model_name=args.llm_model,
            api_keys=api_keys,
            base_url=args.llm_base_url,
            max_concurrent_per_key=5,
        )
        logger.info(f"Caption generation enabled: {args.llm_model}")
    else:
        logger.info("Caption generation disabled (no --llm_model provided)")

    bank = await MemoryBank.build(
        memory_dir=args.memory_dir,
        embedder=embedder,
        api_pool=api_pool,
        filter_correct=not args.no_filter_correct,
        batch_size=args.batch_size,
    )

    logger.info(
        f"Done! Bank has {len(bank.task_ids)} entries, "
        f"embedding dim={bank.embeddings.shape[1]}"
    )


if __name__ == "__main__":
    asyncio.run(main())
