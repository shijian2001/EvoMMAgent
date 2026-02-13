#!/usr/bin/env python3
"""Test 2: Embedder + Reranker vLLM services.

Covers:
  - Embedder.encode_text / encode_batch（调用 /v1/embeddings）
  - MemoryBank.build 离线构建（扫描 → 过滤 → 编码 → 持久化，含 captions.json）
  - 真实 embedding 下的语义搜索
  - Reranker.rerank（调用 /v1/rerank）

Usage:
    python unit_test/trace_level/test_services.py \
        --embedding_model Qwen/Qwen3-VL-Embedding-2B \
        --embedding_base_url http://localhost:8001/v1 \
        --rerank_model Qwen/Qwen3-VL-Reranker-2B \
        --rerank_base_url http://localhost:8002/v1
"""

import argparse
import asyncio
import json
import os
import shutil
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers import (
    FAKE_TRACES, create_fake_memory_dir, cleanup, ok, section,
)

from mm_memory.retrieval.embedder import Embedder
from mm_memory.retrieval.reranker import Reranker
from mm_memory.memory_bank import MemoryBank


async def test_embedder(model: str, base_url: str, api_key: str):
    """验证 Embedder 能正确调用 vLLM /v1/embeddings 接口。"""
    section("1. Embedder")

    embedder = Embedder(model_name=model, base_url=base_url, api_key=api_key)

    # encode_text：2 条文本 → 预期 shape=[2, D], D>0
    embs = await embedder.encode_text(["What color is the car?", "How many people?"])
    assert embs.shape[0] == 2 and embs.shape[1] > 0, f"Unexpected shape: {embs.shape}"
    ok(f"encode_text → shape={embs.shape}")

    # encode_batch：所有 trace 文本，batch_size=2
    texts = [MemoryBank.build_index_text(t) for t in FAKE_TRACES]
    batch_embs = await embedder.encode_batch(texts, batch_size=2)
    assert batch_embs.shape[0] == len(texts), \
        f"Expected {len(texts)} rows, got {batch_embs.shape[0]}"
    ok(f"encode_batch → {batch_embs.shape[0]} texts, dim={batch_embs.shape[1]}")

    return embedder


async def test_memory_bank_build(memory_dir: str, embedder: Embedder):
    """验证 MemoryBank.build 离线构建：扫描 → 过滤 → 编码 → 写文件（含 captions.json）。"""
    section("2. MemoryBank.build (offline)")

    bank_dir = os.path.join(memory_dir, "trace_bank")
    if os.path.exists(bank_dir):
        shutil.rmtree(bank_dir)

    # Build without api_pool → captions all empty
    bank = await MemoryBank.build(
        memory_dir=memory_dir, embedder=embedder,
        filter_correct=True, batch_size=2,
    )
    # 预期：5 条假 trace 中有 4 条 is_correct=True
    assert len(bank.task_ids) == 4, f"Expected 4 entries, got {len(bank.task_ids)}"
    assert os.path.exists(os.path.join(bank_dir, "embeddings.npy")), "embeddings.npy 未生成"
    assert os.path.exists(os.path.join(bank_dir, "task_ids.json")), "task_ids.json 未生成"
    assert os.path.exists(os.path.join(bank_dir, "captions.json")), "captions.json 未生成"

    # Verify captions.json has correct length and all empty (no api_pool)
    with open(os.path.join(bank_dir, "captions.json"), "r") as f:
        captions = json.load(f)
    assert len(captions) == len(bank.task_ids), "captions 长度应与 task_ids 一致"
    assert all(c == "" for c in captions), "无 api_pool 时 captions 应全为空"
    ok(f"Built bank: {len(bank.task_ids)} entries, dim={bank.embeddings.shape[1]}, "
       f"captions.json created (all empty)")

    # 用语义相关的 query 搜索 → car 相关 trace 的 score 应最高
    q_emb = await embedder.encode_text(["What is the color of the vehicle?"])
    results = bank.search(q_emb, top_k=3)
    assert len(results) > 0, "Search returned empty"
    assert all("_caption" in r for r in results), "搜索结果应包含 _caption 字段"
    ok(f"Real search results (expect car-related trace ranked high):")
    for r in results:
        print(f"      {r['task_id']} (score={r['retrieval_score']:.4f}): "
              f"{r['input']['question'][:60]}")

    return bank


async def test_reranker(model: str, base_url: str, api_key: str):
    """验证 Reranker 调用 /v1/rerank，返回按相关性排序的结果。"""
    section("3. Reranker")

    reranker = Reranker(model_name=model, base_url=base_url, api_key=api_key)

    candidates = [
        {"text": "A red car is parked on the street.", "task_id": "a"},
        {"text": "Five people standing in a park.", "task_id": "b"},
        {"text": "The sky is blue and clear.", "task_id": "c"},
    ]
    results = await reranker.rerank("What color is the car?", candidates, top_n=2)

    # 预期：返回 2 条，每条有 rerank_score，分数降序
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    assert all("rerank_score" in r for r in results), "缺少 rerank_score"
    scores = [r["rerank_score"] for r in results]
    assert scores == sorted(scores, reverse=True), "Scores not sorted descending"
    ok(f"Rerank results (expect 'a' ranked first):")
    for r in results:
        print(f"      {r['task_id']} (score={r['rerank_score']:.4f}): {r['text'][:50]}")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_model", required=True,
                        help="e.g. Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--embedding_base_url", required=True,
                        help="e.g. http://localhost:8001/v1")
    parser.add_argument("--embedding_api_key", default="dummy")
    parser.add_argument("--rerank_model", default="",
                        help="e.g. Qwen/Qwen3-VL-Reranker-2B (省略则跳过)")
    parser.add_argument("--rerank_base_url", default="",
                        help="e.g. http://localhost:8002/v1")
    parser.add_argument("--rerank_api_key", default="dummy")
    args = parser.parse_args()

    memory_dir = create_fake_memory_dir()
    try:
        embedder = await test_embedder(
            args.embedding_model, args.embedding_base_url, args.embedding_api_key)
        await test_memory_bank_build(memory_dir, embedder)

        if args.rerank_model and args.rerank_base_url:
            await test_reranker(
                args.rerank_model, args.rerank_base_url, args.rerank_api_key)
        else:
            print("\n  ⏭️  Reranker skipped (no --rerank_model provided)")

        print("\n🎉 test_services ALL PASSED\n")
    finally:
        cleanup(memory_dir)


if __name__ == "__main__":
    asyncio.run(main())
