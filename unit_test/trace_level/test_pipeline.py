#!/usr/bin/env python3
"""Test 3: TracePipeline end-to-end."""

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.async_pool import APIPool
from config import RetrievalConfig
from helpers import cleanup, create_fake_memory_dir, ok, section
from mm_memory.retrieval.embedder import Embedder
from mm_memory.retrieval.trace_pipeline import TracePipeline
from mm_memory.trace_bank import TraceBank
from scripts.build_trace_bank import build_trace_bank


async def test_pipeline(memory_dir: str, api_pool: APIPool, embedder: Embedder):
    section("1. Build trace_bank + pipeline retrieval")

    await build_trace_bank(
        memory_dir=memory_dir,
        api_pool=api_pool,
        embedder=embedder,
        filter_correct=True,
        batch_size=2,
        hindsight_concurrency=2,
        bank_dir_name="trace_bank",
    )
    bank = TraceBank(os.path.join(memory_dir, "trace_bank"))
    ok(f"trace_bank ready: {len(bank.experiences)} entries")

    config = RetrievalConfig(enable=True, bank_memory_dir=memory_dir, trace_top_n=1, min_score=-1.0)
    pipeline = TracePipeline(config=config, trace_bank=bank, embedder=embedder)

    exp = await pipeline.run(
        question="What color is the largest car in the image?",
        images=[],
        sub_task="color_recognition",
    )
    assert isinstance(exp, str) and len(exp) > 0
    ok(f"pipeline.run returns experience ({len(exp)} chars)")

    exp2 = await pipeline.run(
        question="How many people are in the photo?",
        images=[],
        sub_task="counting",
    )
    assert isinstance(exp2, str) and len(exp2) > 0
    ok("second query retrieval also returns non-empty experience")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_model", required=True, help="e.g. Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--embedding_base_url", required=True, help="e.g. http://localhost:8001/v1")
    parser.add_argument("--embedding_api_key", default="dummy")
    parser.add_argument("--llm_model", required=True, help="e.g. qwen3-vl-8b-instruct")
    parser.add_argument("--llm_base_url", required=True, help="e.g. https://maas.devops.xiaohongshu.com/v1")
    parser.add_argument("--llm_api_key", required=True)
    args = parser.parse_args()

    memory_dir = create_fake_memory_dir()
    try:
        api_pool = APIPool(
            model_name=args.llm_model,
            api_keys=[args.llm_api_key],
            base_url=args.llm_base_url,
            max_retries=3,
        )
        embedder = Embedder(
            model_name=args.embedding_model,
            base_url=args.embedding_base_url,
            api_key=args.embedding_api_key,
        )
        await test_pipeline(memory_dir, api_pool, embedder)
        print("\n🎉 test_pipeline ALL PASSED\n")
    finally:
        cleanup(memory_dir)


if __name__ == "__main__":
    asyncio.run(main())
