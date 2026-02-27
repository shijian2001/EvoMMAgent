# Retrieval-Augmented Experience Pipeline

Two retrieval modes, switchable via `config.retrieval.mode`:

| Mode | Granularity | Experience source | Online cost |
|------|-------------|-------------------|-------------|
| `state` | Per-step (MDP state) | Hindsight-annotated (state, action) pairs | 1 VLM caption (first step only) + 1 embedding/step |
| `trace` | Per-task | Summarized from similar traces | LLM rewrite + rerank + summary |

---

## State-Level Retrieval (`mode="state"`)

Agent 推理的每一步，根据当前 state 从历史轨迹中检索最相关的决策经验，注入 conversation。

```
正确 traces ──[offline]──→ MDP trajectory ──→ LLM hindsight annotation
                                                    │
                                            Q-value + experience per (s, a)
                                                    │
                                            embed states → state_bank/
                                                    │
推理时每一步: serialize state → embed → cosine search → inject experience
```

### Offline: 构建 State Bank

```bash
python scripts/build_state_bank.py \
    --memory_dir memory/train_run/ \
    --llm_model qwen3-vl-235b-a22b-instruct \
    --llm_base_url https://maas.devops.xiaohongshu.com/v1 \
    --llm_api_key YOUR_KEY \
    --embedding_model Qwen/Qwen3-VL-Embedding-2B \
    --embedding_base_url http://localhost:8001/v1 \
    --min_q 5
```

流程（全部 in-memory，不修改原始 trace.json）：
1. 扫描 `tasks/*/trace.json`，过滤 `is_correct=True`
2. 转换为 MDP trajectory：`think + action → atomic action a_t = (thinking, tool, params, observation)`，answer 作为终端 action
3. VL 模型 hindsight 标注：一次 LLM 调用/trace，输入完整轨迹 + 原始图片，输出每个 (state, action) 的 Q-value (0-10) 和 experience (1-2 句)
4. VLM 为每条 trace 的输入图像生成 image caption（多图 → 1 caption），复用同一 `api_pool`
5. 过滤 Q >= min_q 的 state，用 `StateBank.state_to_text(image_caption=...)` 序列化
6. Embedding 编码 → 持久化 `state_bank/embeddings.npy` + `state_meta.json`（含 `image_caption` 字段）

### Online: 每步检索

1. （首步）VLM 生成 image caption 并缓存（1 次调用，后续复用）
2. 每轮 `StateBank.state_to_text(image_caption=cached)` 序列化当前 state（与离线同一函数，格式一致）
3. Embed → cosine search StateBank（Q-value 二次过滤）
4. 取 top `experience_top_n` 条预计算 experience，注入为 user message
5. 下一轮开始前清除旧 experience message

**无 rerank，无 summary LLM 调用，延迟极低。**

### 配置项

| 配置项 | 默认 | 说明 |
|-------|------|------|
| `mode` | `"state"` | 设为 `"state"` 启用 |
| `min_q_value` | `5` | Q-value 过滤阈值 |
| `experience_top_n` | `1` | 每步注入的 experience 条数 |
| `retrieval_top_k` | `10` | cosine 候选数（过滤前） |
| `min_score` | `0.1` | cosine 相似度阈值 |

---

## Trace-Level Retrieval (`mode="trace"`)

Agent 推理前，从历史成功 trace 中检索相似任务的经验，注入 system prompt。

```
正确 traces ──[offline build]──→ Trace Bank (embeddings + captions)
                                        │
新 task (question, images) ──────→ TracePipeline ──→ experience string
                                        │
                                  agent system prompt ← "## Experience from Similar Tasks"
```

### Offline: 构建 Trace Bank

```bash
python scripts/build_memory_bank.py \
    --memory_dir memory/train_run/ \
    --embedding_model Qwen/Qwen3-VL-Embedding-2B \
    --embedding_base_url http://localhost:8001/v1 \
    --llm_model qwen3-vl-32b-instruct \
    --llm_base_url https://... --llm_api_key ...
```

流程：
1. 扫描 `tasks/*/trace.json`，过滤 `is_correct=True`
2. （可选）VLM 为每条 trace 的输入图像生成 caption
3. 构建 index text = `Image description: {caption}\nQuestion: {question}\nTask: {sub_task}\nTools: {tools}`
4. Embedding 编码 → 持久化 `trace_bank/embeddings.npy` + `task_ids.json` + `captions.json`

### Online: 检索流程

```
① Query Rewrite (LLM) → text_queries + image_caption
② Embed + cosine search trace_bank
③ Deduplicate by task_id
④ Rerank (top_n)
⑤ Summary (LLM) → experience string
⑥ (多轮) Sufficiency check → 下一轮 or return
```

### 配置项

| 配置项 | 默认 | 说明 |
|-------|------|------|
| `mode` | — | 设为 `"trace"` 启用 |
| `enable_rerank` | `True` | 关闭后按 retrieval_score 排序 |
| `enable_query_rewrite` | `True` | 关闭后直接用原始 question |
| `max_retrieval_rounds` | `1` | 1=单轮，2+=多轮 |

---

## 文件结构

```
mm_memory/
├── memory_bank.py              # MemoryBank: trace-level 索引 (trace_bank/)
├── state_bank.py               # StateBank: state-level 索引 (state_bank/)
└── retrieval/
    ├── __init__.py
    ├── trace_pipeline.py       # TracePipeline: trace-level 流程编排
    ├── query_rewriter.py       # QueryRewriter: LLM 多模态改写 (trace mode)
    ├── embedder.py             # Embedder: vLLM /v1/embeddings (shared)
    └── reranker.py             # Reranker: vLLM /v1/rerank (trace mode)
scripts/
├── build_state_bank.py         # State-level offline 构建
└── build_memory_bank.py        # Trace-level offline 构建
```
