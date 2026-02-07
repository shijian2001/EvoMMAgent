#!/usr/bin/env python3
"""Test 1: Pure local logic — no network required.

Covers:
  - RetrievalConfig / Config 默认值
  - MemoryBank.build_index_text 索引文本构建
  - MemoryBank 加载、cosine 搜索、trace 按需加载
  - MemoryBank 缺失时的错误处理

Usage:
    python unit_test/test_local.py
"""

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
    ok("RetrievalConfig defaults correct (enable=False)")

    cfg = Config()
    assert cfg.retrieval.enable is False, "Config 集成后仍应默认关闭"
    ok("Config.retrieval integrated, disabled by default")


def test_build_index_text():
    """验证 trace → 索引文本的转换：应包含 question、tools、answer。"""
    section("2. build_index_text")

    # 第一条 trace：color_recognition，用了 localize_objects，答案 Red
    text = MemoryBank.build_index_text(FAKE_TRACES[0])
    assert "What color" in text, "应包含 question"
    assert "localize_objects" in text, "应包含工具名"
    assert "Red" in text, "应包含 answer"
    ok(f"Trace 000001 → '{text[:80]}...'")

    # 所有 trace 都应产生非空文本
    for t in FAKE_TRACES:
        txt = MemoryBank.build_index_text(t)
        assert len(txt) > 0, f"Trace {t['task_id']} 生成了空文本"
    ok(f"All {len(FAKE_TRACES)} traces produce non-empty index text")


def test_memory_bank_search():
    """验证 MemoryBank 能加载 bank/ 文件，执行 cosine search，并按需加载 trace。"""
    section("3. MemoryBank Load & Search")

    memory_dir = create_fake_memory_dir()
    try:
        # 写入随机 embedding（dim=8）模拟已构建的 bank
        create_synthetic_bank(memory_dir, dim=8)

        bank = MemoryBank(memory_dir)
        # 预期：只有 3 条 correct trace
        assert len(bank.task_ids) == len(CORRECT_TRACES), \
            f"Expected {len(CORRECT_TRACES)} entries, got {len(bank.task_ids)}"
        ok(f"Loaded {len(bank.task_ids)} entries, dim={bank.embeddings.shape[1]}")

        # cosine search
        query = np.random.randn(1, 8).astype(np.float32)
        results = bank.search(query, top_k=2)

        # 预期：返回 ≤2 条结果，每条有 retrieval_score 和完整 trace 数据
        assert len(results) <= 2
        assert all("retrieval_score" in r for r in results), "缺少 retrieval_score"
        assert all("input" in r for r in results), "trace 数据未按需加载"
        ok(f"Search → {len(results)} results: "
           f"{[(r['task_id'], round(r['retrieval_score'], 4)) for r in results]}")
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


if __name__ == "__main__":
    test_config_defaults()
    test_build_index_text()
    test_memory_bank_search()
    test_memory_bank_missing()
    print("\n🎉 test_local ALL PASSED\n")
