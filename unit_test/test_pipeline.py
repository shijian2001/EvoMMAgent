#!/usr/bin/env python3
"""Test 3: QueryRewriter + Full RetrievalPipeline.

Covers:
  - QueryRewriter.rewrite（text_only 策略，带/不带上轮 context）
  - RetrievalPipeline.run 四种组合：
    a) 单轮 + 全功能（rewrite + rerank）
    b) 多轮 (2 rounds) + sufficiency 判断
    c) 关闭 rewrite
    d) 关闭 rerank

Usage:
    python unit_test/test_pipeline.py \
        --embedding_model Qwen/Qwen3-VL-Embedding-2B \
        --embedding_base_url http://localhost:8001/v1 \
        --rerank_model Qwen/Qwen3-VL-Reranker-2B \
        --rerank_base_url http://localhost:8002/v1 \
        --llm_model qwen3-vl-8b-instruct \
        --llm_base_url https://maas.devops.xiaohongshu.com/v1 \
        --llm_api_key YOUR_API_KEY
"""

import argparse
import asyncio
import os
import shutil

from helpers import create_fake_memory_dir, cleanup, ok, section

from config import RetrievalConfig
from api.async_pool import APIPool
from mm_memory.memory_bank import MemoryBank
from mm_memory.retrieval.embedder import Embedder
from mm_memory.retrieval.reranker import Reranker
from mm_memory.retrieval.query_rewriter import QueryRewriter
from mm_memory.retrieval.pipeline import RetrievalPipeline


# ── QueryRewriter ────────────────────────────────────────────────────────────

async def test_query_rewriter(api_pool: APIPool):
    """验证 LLM 能生成改写 query，并保持原始 question 为第一条。"""
    section("1. QueryRewriter")

    rewriter = QueryRewriter(api_pool=api_pool, max_sub_queries=3)

    # 基础改写：预期返回 ≥2 条 query（原始 + 至少 1 条改写）
    result = await rewriter.rewrite(
        question="What color is the largest car in the image?",
        strategy="text_only",
    )
    assert result["text_queries"][0] == "What color is the largest car in the image?", \
        "第一条应是原始 question"
    assert len(result["text_queries"]) > 1, "应至少生成 1 条改写 query"
    ok(f"Basic rewrite → {len(result['text_queries'])} queries:")
    for i, q in enumerate(result["text_queries"]):
        print(f"      [{i}] {q}")

    # 带上轮 context 的改写：模拟多轮场景，LLM 应基于 context 调整方向
    result2 = await rewriter.rewrite(
        question="What color is the largest car in the image?",
        strategy="text_only",
        previous_context="Found depth estimation tasks. Still need color-related tasks.",
    )
    assert len(result2["text_queries"]) > 1
    ok(f"Rewrite with context → {len(result2['text_queries'])} queries")

    return rewriter


# ── Full Pipeline ────────────────────────────────────────────────────────────

async def test_pipeline(
    memory_dir: str,
    embedder: Embedder, reranker: Reranker,
    api_pool: APIPool, rewriter: QueryRewriter,
):
    """验证 RetrievalPipeline.run 在不同配置下都能正常返回 experience。"""
    section("2. Full Pipeline")

    # 先用真实 embedding 构建 bank
    bank_dir = os.path.join(memory_dir, "bank")
    if os.path.exists(bank_dir):
        shutil.rmtree(bank_dir)
    bank = await MemoryBank.build(
        memory_dir=memory_dir, embedder=embedder,
        filter_correct=True, batch_size=2,
    )
    ok(f"Bank ready: {len(bank.task_ids)} entries")

    # ── 2a. 单轮 + 全功能 ──
    # 链路: rewrite → embed queries → search bank → rerank → LLM summary
    # 预期: 返回 2-3 句经验总结文本
    config_a = RetrievalConfig(
        enable=True, bank_memory_dir=memory_dir,
        enable_query_rewrite=True, max_sub_queries=2,
        retrieval_top_k=3, enable_rerank=True, rerank_top_n=2,
        max_retrieval_rounds=1,
    )
    pipeline_a = RetrievalPipeline(
        config=config_a, memory_bank=bank,
        embedder=embedder, reranker=reranker,
        api_pool=api_pool, query_rewriter=rewriter,
    )
    exp_a = await pipeline_a.run("What color is the biggest vehicle in the photo?")
    assert isinstance(exp_a, str) and len(exp_a) > 0, "experience 不应为空"
    ok(f"[a] Single-round full: {len(exp_a)} chars")
    print(f"      \"{exp_a[:150]}...\"")

    # ── 2b. 多轮 (2 rounds) ──
    # 链路: round1(rewrite→search→rerank) → sufficiency judge → 若不足则 round2 → summary
    # 预期: 返回非空 experience，日志可见 sufficiency 判断
    config_b = RetrievalConfig(
        enable=True, bank_memory_dir=memory_dir,
        enable_query_rewrite=True, max_sub_queries=2,
        retrieval_top_k=3, enable_rerank=True, rerank_top_n=2,
        max_retrieval_rounds=2,
    )
    pipeline_b = RetrievalPipeline(
        config=config_b, memory_bank=bank,
        embedder=embedder, reranker=reranker,
        api_pool=api_pool, query_rewriter=rewriter,
    )
    exp_b = await pipeline_b.run("How many animals and which is closer to the camera?")
    assert isinstance(exp_b, str) and len(exp_b) > 0
    ok(f"[b] Multi-round (2): {len(exp_b)} chars")

    # ── 2c. 关闭 rewrite ──
    # 链路: 直接用原始 question embed → search → rerank → summary
    # 预期: 正常返回 experience（只是少了改写的多角度查询）
    config_c = RetrievalConfig(
        enable=True, bank_memory_dir=memory_dir,
        enable_query_rewrite=False,
        retrieval_top_k=3, enable_rerank=True, rerank_top_n=2,
        max_retrieval_rounds=1,
    )
    pipeline_c = RetrievalPipeline(
        config=config_c, memory_bank=bank,
        embedder=embedder, reranker=reranker,
        api_pool=api_pool, query_rewriter=None,
    )
    exp_c = await pipeline_c.run("How many people are in the image?")
    assert isinstance(exp_c, str)
    ok(f"[c] No rewrite: {len(exp_c)} chars")

    # ── 2d. 关闭 rerank ──
    # 链路: rewrite → embed → search → 按 retrieval_score 排序取 top-n → summary
    # 预期: 正常返回 experience（跳过 reranker，用原始检索分数排序）
    config_d = RetrievalConfig(
        enable=True, bank_memory_dir=memory_dir,
        enable_query_rewrite=True, max_sub_queries=2,
        retrieval_top_k=3, enable_rerank=False, rerank_top_n=2,
        max_retrieval_rounds=1,
    )
    pipeline_d = RetrievalPipeline(
        config=config_d, memory_bank=bank,
        embedder=embedder, reranker=None,
        api_pool=api_pool, query_rewriter=rewriter,
    )
    exp_d = await pipeline_d.run("What color is the car?")
    assert isinstance(exp_d, str)
    ok(f"[d] No rerank: {len(exp_d)} chars")


# ── Main ─────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_model", required=True,
                        help="e.g. Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--embedding_base_url", required=True,
                        help="e.g. http://localhost:8001/v1")
    parser.add_argument("--embedding_api_key", default="dummy")
    parser.add_argument("--rerank_model", required=True,
                        help="e.g. Qwen/Qwen3-VL-Reranker-2B")
    parser.add_argument("--rerank_base_url", required=True,
                        help="e.g. http://localhost:8002/v1")
    parser.add_argument("--rerank_api_key", default="dummy")
    parser.add_argument("--llm_model", required=True,
                        help="e.g. qwen3-vl-8b-instruct")
    parser.add_argument("--llm_base_url", required=True,
                        help="e.g. https://maas.devops.xiaohongshu.com/v1")
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
        reranker = Reranker(
            model_name=args.rerank_model,
            base_url=args.rerank_base_url,
            api_key=args.rerank_api_key,
        )

        rewriter = await test_query_rewriter(api_pool)
        await test_pipeline(memory_dir, embedder, reranker, api_pool, rewriter)

        print("\n🎉 test_pipeline ALL PASSED\n")
    finally:
        cleanup(memory_dir)


if __name__ == "__main__":
    asyncio.run(main())
