# Retrieval-Augmented Experience Pipeline

## Overview

Agent 在推理前，从历史成功 trace 中检索相似任务的经验，注入 system prompt 指导当前任务。

```
训练集 traces ──[offline build]──→ Memory Bank (embeddings + captions)
                                        │
新 task (question, images) ──────→ Retrieval Pipeline ──→ experience string
                                        │
                                  agent system prompt ← "## Experience from Similar Tasks"
```

## Offline: 构建 Memory Bank

```bash
python scripts/build_memory_bank.py \
    --memory_dir memory/train_run/ \
    --embedding_model Qwen/Qwen3-VL-Embedding-2B \
    --embedding_base_url http://localhost:8001/v1 \
    --llm_model qwen3-vl-32b-instruct \          # 可选，生成图像 caption
    --llm_base_url https://... --llm_api_key ...
```

流程：
1. 扫描 `tasks/*/trace.json`，过滤 `is_correct=True`
2. （可选）VLM 为每条 trace 的输入图像生成 caption（并发 sem=8）
3. 构建 index text = `Image description: {caption}\n{question}\nTask: {sub_task}\nTools (in order): {tools}`
4. Embedding 模型编码所有 index text
5. 持久化 → `bank/embeddings.npy` + `task_ids.json` + `captions.json`

## Online: 检索流程（每个 task）

```
┌───────────────────── Round N (每轮独立) ─────────────────────┐
│                                                              │
│  ① Query Rewrite (LLM, multimodal)                          │
│     input:  question + images + search_context(多轮)         │
│     output: text_queries[], image_caption                    │
│                                                              │
│  ② Embed + Search                                           │
│     - encode_text(text_queries) → 一次 batch API call        │
│     - 逐 query 搜索 bank (cosine, min_score 过滤)            │
│     - 每轮独立搜索，不与上轮混合                                │
│                                                              │
│  ③ Deduplicate                                              │
│     - 按 task_id 去重，保留最高 retrieval_score               │
│                                                              │
│  ④ Rerank                                                   │
│     - query = question + image_caption                       │
│     - document = build_index_text(trace, caption)            │
│     - 取 top_n                                               │
│                                                              │
│  ⑤ Summary (LLM)                                            │
│     - 输入：question + image_caption + candidates 推理链      │
│     - 输出：actionable experience (几句话)                    │
│                                                              │
│  ⑥ Sufficiency Check (LLM, 仅多轮)                          │
│     - 判断 experience 是否足够                                │
│     - YES → return experience                                │
│     - NO  → search_context = summary + gap → 下一轮 ①        │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## 注入 Agent

experience 通过 Jinja template 注入 system prompt：

```
## Experience from Similar Tasks
The following experience is derived from similar previously solved tasks.
Use it as reference, not as strict rules.

{experience}
```

experience 为空时不渲染该 section。Pipeline 任何异常均 fallback 到空 experience，不影响 agent 正常执行。

## 可插拔开关

| 配置项 | 默认 | 说明 |
|-------|------|------|
| `enable` | `False` | 总开关，关闭时完全不初始化 |
| `enable_query_rewrite` | `True` | 关闭后直接用原始 question 检索 |
| `enable_rerank` | `True` | 关闭后按 retrieval_score 排序取 top_n |
| `max_retrieval_rounds` | `1` | 1=单轮，2+=多轮 deep research |
| `min_score` | `0.1` | cosine 相似度阈值，低于此分的 trace 不返回 |

## 文件结构

```
mm_memory/
├── memory_bank.py           # MemoryBank: 索引加载、搜索、offline build
└── retrieval/
    ├── __init__.py
    ├── pipeline.py           # RetrievalPipeline: 流程编排
    ├── query_rewriter.py     # QueryRewriter: LLM 多模态改写
    ├── embedder.py           # Embedder: vLLM /v1/embeddings
    └── reranker.py           # Reranker: vLLM /v1/rerank
scripts/
└── build_memory_bank.py      # Offline 构建脚本
```

## TODO

- **推理时增量扩展 Bank**：agent 推理成功后，新 trace 自动追加进 bank。设计为异步后台 rebuild（不阻塞 agent），可按时间间隔或累积条数触发。当前架构已支持（Memory 系统自动保存 trace，`build_memory_bank.py` 可重复执行），只需增加触发机制。
- **Bank 规模扩展**：当前全量 cosine 扫描，万级 trace 无压力。若扩展到十万级以上，考虑换 FAISS 近似最近邻。
