# Retrieval-Augmented Experience Pipeline

系统支持两种检索模式，由 `config.retrieval.mode` 决定：

- `state`: state-level（主方法，不变）
- `trace`: 简化后的 trace-level baseline（本文重点）

---

## Trace-Level

### 目标

在 agent 推理开始前，检索 1 条最相关历史经验，注入 system prompt：

1. **离线**：每条正确 trace 用 VLM hindsight 生成 `1-2` 句经验
2. **离线**：对 `(images + Question + Task)` 做多模态 embedding
3. **在线**：对当前任务 `(images + Question + Task)` 做多模态 embedding
4. **在线**：余弦 `top-1` 检索并返回对应经验

在线链路不再包含 query rewrite / rerank / summary / multi-round。

---

## Offline: 构建 Trace Bank

```bash
python scripts/build_trace_bank.py \
    --memory_dir memory/train_run/ \
    --llm_model qwen-vl-hindsight \
    --llm_base_url http://localhost:8003/v1 \
    --llm_api_key YOUR_KEY \
    --embedding_model Qwen/Qwen3-VL-Embedding-2B \
    --embedding_base_url http://localhost:8001/v1
```

产物目录：

```text
{memory_dir}/trace_bank/
├── embeddings.npy
└── experiences.json
```

---

## Online: 检索流程

`TracePipeline.run(question, images, sub_task)`:

1. 组装 query text: `Question: ...\nTask: ...`
2. `encode_multimodal(query_text, images)`（若无图则 text embedding）
3. `TraceBank.search` 余弦 top-1
4. 返回 experience 字符串，注入 system prompt

---

## 相关代码

```text
mm_memory/
├── trace_bank.py               # TraceBank: 载入 embeddings+experiences，top-1 检索
└── retrieval/
    ├── embedder.py             # 多模态/文本 embedding
    └── trace_pipeline.py       # 简化 trace 检索流程

scripts/
└── build_trace_bank.py         # trace-level offline 构建
```

---

## State-Level 说明

state-level 代码与流程未改动，继续使用原有：

- `mm_memory/state_bank.py`
- `scripts/build_state_bank.py`
- `tool/search_experiences_tool.py`
