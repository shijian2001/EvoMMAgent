#!/usr/bin/env python3
"""Test 1: Pure local logic for trace-level retrieval."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, RetrievalConfig
from helpers import CORRECT_TRACES, FAKE_TRACES, cleanup, create_fake_memory_dir, create_synthetic_bank, ok, section
from mm_memory.trace_bank import TraceBank


def test_config_defaults():
    section("1. Config Defaults")
    rc = RetrievalConfig()
    assert rc.enable is False
    assert rc.trace_top_n == 1
    assert rc.min_score == 0.1
    ok("RetrievalConfig defaults match trace mode")

    cfg = Config()
    assert cfg.retrieval.enable is False
    ok("Config.retrieval integrated, disabled by default")


def test_build_index_text():
    section("2. TraceBank.build_index_text")
    text = TraceBank.build_index_text(FAKE_TRACES[0])
    assert "Question: What color is the largest car" in text
    assert "Task: color_recognition" in text
    assert "Tools" not in text
    assert "Answer" not in text
    ok("index text only contains Question + Task")

    for t in FAKE_TRACES:
        txt = TraceBank.build_index_text(t)
        assert len(txt) > 0
        assert "Answer" not in txt
    ok(f"All {len(FAKE_TRACES)} traces produce valid index text")


def test_trace_bank_search():
    section("3. TraceBank load + top1 search")
    memory_dir = create_fake_memory_dir()
    try:
        create_synthetic_bank(memory_dir, dim=8)
        bank = TraceBank(os.path.join(memory_dir, "trace_bank"))
        assert len(bank.experiences) == len(CORRECT_TRACES)
        ok(f"Loaded {len(bank.experiences)} entries, dim={bank.embeddings.shape[1]}")

        query = np.random.randn(1, 8).astype(np.float32)
        exp = bank.search(query, min_score=-1.0)
        assert isinstance(exp, str) and exp
        ok(f"Top1 retrieval returns one experience: {exp}")

        none_exp = bank.search(query, min_score=1.1)
        assert none_exp is None
        ok("min_score filter works for top1")
    finally:
        cleanup(memory_dir)


def test_trace_bank_missing():
    section("4. TraceBank missing bank")
    memory_dir = create_fake_memory_dir()
    try:
        raised = False
        try:
            TraceBank(os.path.join(memory_dir, "trace_bank"))
        except FileNotFoundError:
            raised = True
        assert raised
        ok("FileNotFoundError raised when trace_bank missing")
    finally:
        cleanup(memory_dir)


if __name__ == "__main__":
    test_config_defaults()
    test_build_index_text()
    test_trace_bank_search()
    test_trace_bank_missing()
    print("\n🎉 test_local ALL PASSED\n")
