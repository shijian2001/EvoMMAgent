#!/usr/bin/env python3
"""Test 1: Pure local logic — no network required.

Covers:
  - RetrievalConfig / Config 默认值
  - MemoryBank.build_index_text 索引文本构建（caption、tools 去重、answer 提取）
  - MemoryBank 加载、cosine 搜索、trace 按需加载、caption 传递
  - MemoryBank 缺失时的错误处理

Usage:
    python unit_test/trace_level/test_local.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from helpers import (
    FAKE_TRACES, CORRECT_TRACES,
    create_fake_memory_dir, create_synthetic_bank, cleanup, ok, section,
)

from config import RetrievalConfig, Config
from mm_memory.memory_bank import MemoryBank


def test_config_defaults():
    """验证新加的 RetrievalConfig 默认全部关闭，不影响原有流程。"""
    section("1. Config Defaults")

    rc = RetrievalConfig()
    assert rc.enable is False, "retrieval 应默认关闭"
    assert rc.max_retrieval_rounds == 1
    assert rc.enable_rerank is True
    assert rc.min_score == 0.1, f"min_score default should be 0.1, got {rc.min_score}"
    assert not hasattr(rc, "query_rewrite_strategy"), \
        "query_rewrite_strategy should have been removed"
    ok("RetrievalConfig defaults correct (enable=False, min_score=0.1)")

    cfg = Config()
    assert cfg.retrieval.enable is False, "Config 集成后仍应默认关闭"
    ok("Config.retrieval integrated, disabled by default")


def test_build_index_text():
    """验证 trace → 索引文本：caption、tools 去重有序、不含 answer。"""
    section("2. build_index_text")

    # ── 2a. 基本 trace（无 caption）──
    text = MemoryBank.build_index_text(FAKE_TRACES[0])
    assert "What color" in text, "应包含 question"
    assert "localize_objects" in text, "应包含工具名"
    assert "Answer" not in text, "index text 不应包含 answer"
    ok(f"Trace 000001 → '{text[:80]}...'")

    # ── 2b. 带 caption ──
    text_cap = MemoryBank.build_index_text(FAKE_TRACES[0], caption="A red car in a parking lot")
    assert text_cap.startswith("Image description: A red car"), "caption 应以 'Image description:' 开头"
    ok(f"With caption → '{text_cap[:80]}...'")

    # ── 2c. Tools 去重保序 ──
    # Trace 000005 has: localize_objects, zoom_in, zoom_in, zoom_in
    trace_005 = FAKE_TRACES[4]
    text_005 = MemoryBank.build_index_text(trace_005)
    assert "Tools (in order): localize_objects, zoom_in" in text_005, \
        f"Tools 应去重保序，实际: {text_005}"
    # 不应有重复
    assert text_005.count("zoom_in") == 1, "zoom_in 应只出现一次"
    ok(f"Tools dedup → '{[l for l in text_005.splitlines() if 'Tools' in l][0]}'")

    # ── 2d. Answer 不应出现 ──
    assert "Answer" not in text_005, "index text 不应包含 answer"
    assert "decreasing geometric" not in text_005, "分析文本不应出现在 index text 中"
    ok("No answer in index text (retrieval targets strategy, not answer)")

    # ── 2e. 所有 trace 都应产生非空文本 ──
    for t in FAKE_TRACES:
        txt = MemoryBank.build_index_text(t)
        assert len(txt) > 0, f"Trace {t['task_id']} 生成了空文本"
        assert "Answer" not in txt, f"Trace {t['task_id']} 不应含 answer"
    ok(f"All {len(FAKE_TRACES)} traces produce non-empty index text without answer")


def test_memory_bank_search():
    """验证 MemoryBank 能加载 bank/ 文件，执行 cosine search，结果含 _caption。"""
    section("3. MemoryBank Load & Search")

    memory_dir = create_fake_memory_dir()
    try:
        # 写入随机 embedding（dim=8）模拟已构建的 bank
        create_synthetic_bank(memory_dir, dim=8)

        bank = MemoryBank(memory_dir)
        assert len(bank.task_ids) == len(CORRECT_TRACES), \
            f"Expected {len(CORRECT_TRACES)} entries, got {len(bank.task_ids)}"
        ok(f"Loaded {len(bank.task_ids)} entries, dim={bank.embeddings.shape[1]}")

        # cosine search
        query = np.random.randn(1, 8).astype(np.float32)
        results = bank.search(query, top_k=2)

        assert len(results) <= 2
        assert all("retrieval_score" in r for r in results), "缺少 retrieval_score"
        assert all("_caption" in r for r in results), "缺少 _caption 字段"
        assert all("input" in r for r in results), "trace 数据未按需加载"
        ok(f"Search → {len(results)} results: "
           f"{[(r['task_id'], round(r['retrieval_score'], 4)) for r in results]}")

        # min_score filtering: very high threshold should return fewer or zero results
        results_high = bank.search(query, top_k=10, min_score=0.99)
        assert all(r["retrieval_score"] >= 0.99 for r in results_high), \
            "min_score 过滤失效：存在低于阈值的结果"
        ok(f"min_score=0.99 → {len(results_high)} results (filtered)")
    finally:
        cleanup(memory_dir)


def test_memory_bank_missing():
    """验证 bank/ 不存在时抛出 FileNotFoundError。"""
    section("4. MemoryBank Error Handling")

    memory_dir = create_fake_memory_dir()
    try:
        raised = False
        try:
            MemoryBank(memory_dir)  # 没有 bank/ 目录
        except FileNotFoundError:
            raised = True
        assert raised, "应抛出 FileNotFoundError"
        ok("FileNotFoundError raised when bank/ missing")
    finally:
        cleanup(memory_dir)


def test_deduplicate():
    """验证 _deduplicate 保留最高 retrieval_score 的候选。"""
    section("5. Pipeline._deduplicate")

    from mm_memory.retrieval.trace_pipeline import TracePipeline

    # 用 None 占位，只测 _deduplicate 纯函数
    pipeline = TracePipeline(
        config=RetrievalConfig(), memory_bank=None,
        embedder=None, reranker=None, api_pool=None,
    )

    candidates = [
        {"task_id": "A", "retrieval_score": 0.5, "data": "first"},
        {"task_id": "B", "retrieval_score": 0.8, "data": "B"},
        {"task_id": "A", "retrieval_score": 0.9, "data": "second"},  # 同 A，更高分
        {"task_id": "A", "retrieval_score": 0.3, "data": "third"},   # 同 A，更低分
    ]
    result = pipeline._deduplicate(candidates)
    ids = {c["task_id"] for c in result}
    assert ids == {"A", "B"}, f"应只剩 A, B，实际: {ids}"
    a_entry = [c for c in result if c["task_id"] == "A"][0]
    assert a_entry["retrieval_score"] == 0.9, "A 应保留 score=0.9 的那条"
    assert a_entry["data"] == "second", "A 应保留 data='second' 的那条"
    ok("Deduplicate keeps highest score per task_id")


def test_format_candidates():
    """验证 _format_candidates 格式：think/answer 内容、[tool]、skip observation。"""
    section("6. Pipeline._format_candidates")

    from mm_memory.retrieval.trace_pipeline import TracePipeline

    pipeline = TracePipeline(
        config=RetrievalConfig(), memory_bank=None,
        embedder=None, reranker=None, api_pool=None,
    )

    # 用 FAKE_TRACES[0]: think → action → observation → think
    candidates = [FAKE_TRACES[0].copy()]
    candidates[0]["_caption"] = "A parking lot with cars"

    text = pipeline._format_candidates(candidates)

    # 应包含 think 内容
    assert "I need to find the largest car first" in text, "应包含 think content"
    # 应包含 [tool_name] 而非 tool 的原始 dict
    assert "[localize_objects]" in text, "action 应显示为 [tool_name]"
    # 不应包含 observation 内容
    assert "Found 3 cars" not in text, "observation 应被 skip"
    # 应包含 caption
    assert "Image description: A parking lot" in text, "应包含 caption"
    # 应包含 question
    assert "What color" in text, "应包含 question"
    ok("Format: think content + [tool] + skip observation + caption + question")

    # 无 caption 时不应有 Image description 行
    candidates_no_cap = [FAKE_TRACES[1].copy()]
    candidates_no_cap[0]["_caption"] = ""
    text2 = pipeline._format_candidates(candidates_no_cap)
    assert "Image description" not in text2, "无 caption 时不应出现 Image description"
    ok("No caption → no Image description line")


if __name__ == "__main__":
    test_config_defaults()
    test_build_index_text()
    test_memory_bank_search()
    test_memory_bank_missing()
    test_deduplicate()
    test_format_candidates()
    print("\n🎉 test_local ALL PASSED\n")
