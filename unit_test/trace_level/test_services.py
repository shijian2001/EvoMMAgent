#!/usr/bin/env python3
"""Test 2: Embedder + TraceBank services."""

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.async_pool import APIPool
from helpers import FAKE_TRACES, cleanup, create_fake_memory_dir, ok, section
from mm_memory.retrieval.embedder import Embedder
from mm_memory.trace_bank import TraceBank
from scripts.build_trace_bank import build_trace_bank


async def test_embedder(model: str, base_url: str, api_key: str):
    section("1. Embedder")
    embedder = Embedder(model_name=model, base_url=base_url, api_key=api_key)

    embs = await embedder.encode_text(["What color is the car?", "How many people?"])
    assert embs.shape[0] == 2 and embs.shape[1] > 0
    ok(f"encode_text -> shape={embs.shape}")

    payloads = []
    for t in FAKE_TRACES[:3]:
        payloads.append({"text": TraceBank.build_index_text(t), "images": []})
    batch_embs = await embedder.encode_multimodal_batch(payloads, batch_size=2)
    assert batch_embs.shape[0] == len(payloads)
    ok(f"encode_multimodal_batch -> {batch_embs.shape[0]} payloads, dim={batch_embs.shape[1]}")

    return embedder


async def test_trace_bank_build(memory_dir: str, api_pool: APIPool, embedder: Embedder):
    section("2. build_trace_bank + TraceBank search")
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
    assert os.path.exists(os.path.join(memory_dir, "trace_bank", "embeddings.npy"))
    assert os.path.exists(os.path.join(memory_dir, "trace_bank", "experiences.json"))
    assert len(bank.experiences) > 0
    ok(f"trace_bank built: {len(bank.experiences)} entries")

    q_emb = await embedder.encode_text(["Question: What is the color of the vehicle?\nTask: color_recognition"])
    exp = bank.search(q_emb, min_score=-1.0)
    assert isinstance(exp, str) and exp
    ok("TraceBank top1 search returns one non-empty experience")


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
        embedder = await test_embedder(
            args.embedding_model, args.embedding_base_url, args.embedding_api_key
        )
        api_pool = APIPool(
            model_name=args.llm_model,
            api_keys=[args.llm_api_key],
            base_url=args.llm_base_url,
            max_retries=3,
        )
        await test_trace_bank_build(memory_dir, api_pool, embedder)
        print("\n🎉 test_services ALL PASSED\n")
    finally:
        cleanup(memory_dir)


if __name__ == "__main__":
    asyncio.run(main())
