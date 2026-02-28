# Retrieval Pipeline 测试指南

## 一、vLLM 服务部署

### 1. Embedding 服务 (port 8001)

```bash
vllm serve /Path/to/Qwen3-VL-Embedding-8B \
    --runner pooling \
    --port 8001 \
    --dtype float16 \
    --trust-remote-code \
    --served-model-name qwen3vl-embed
```

### 2. LLM/VLM 服务 (用于 hindsight，port 自定义)

```bash
vllm serve /Path/to/Qwen2.5-VL-7B-Instruct \
    --port 8003 \
    --dtype float16 \
    --served-model-name qwen-vl-hindsight
```

---

## 二、测试目录结构

```text
unit_test/
├── helpers.py
├── test_answer_match.py
├── README.md
├── trace_level/
│   ├── test_local.py      # TraceBank + config 本地逻辑
│   ├── test_services.py   # Embedder + build_trace_bank + TraceBank
│   └── test_pipeline.py   # 简化 TracePipeline 端到端
└── state_level/
    ├── test_local.py
    ├── test_pipeline.py
    └── test_agent_integration.py
```

---

## 三、Trace-Level 测试

### test_local.py — 纯本地逻辑

```bash
python unit_test/trace_level/test_local.py
```

覆盖：
- `RetrievalConfig` 默认值（`trace_top_n=1`）
- `TraceBank.build_index_text`（仅 `Question + Task`）
- `TraceBank` 加载 + top-1 搜索

### test_services.py — Embedder + TraceBank 构建（需要 Embedding + LLM/VLM）

```bash
python unit_test/trace_level/test_services.py \
    --embedding_model Qwen/Qwen3-VL-Embedding-2B \
    --embedding_base_url http://localhost:8001/v1 \
    --llm_model qwen-vl-hindsight \
    --llm_base_url http://localhost:8003/v1 \
    --llm_api_key YOUR_API_KEY
```

覆盖：
- `Embedder.encode_text` / `encode_multimodal_batch`
- `scripts/build_trace_bank.py` 产物 (`embeddings.npy`, `experiences.json`)
- `TraceBank.search` top-1 返回

### test_pipeline.py — 简化 TracePipeline 端到端（需要 Embedding + LLM/VLM）

```bash
python unit_test/trace_level/test_pipeline.py \
    --embedding_model Qwen/Qwen3-VL-Embedding-2B \
    --embedding_base_url http://localhost:8001/v1 \
    --llm_model qwen-vl-hindsight \
    --llm_base_url http://localhost:8003/v1 \
    --llm_api_key YOUR_API_KEY
```

覆盖：
- 离线构建 trace_bank
- 在线 `encode_multimodal -> cosine top-1 -> experience` 链路

---

## 四、State-Level 测试

### test_local.py — 纯本地逻辑（无需服务）

```bash
python unit_test/state_level/test_local.py
```

覆盖：
- `RetrievalConfig` 的 state 默认配置
- `StateBank` 本地加载与检索行为（含阈值过滤）
- `SearchExperiencesTool` 的 round/use 限制逻辑

### test_pipeline.py — state-level 端到端（需要 Embedding 服务）

```bash
python unit_test/state_level/test_pipeline.py \
    --embedding_model Qwen/Qwen3-VL-Embedding-2B \
    --embedding_base_url http://localhost:8001/v1
```

覆盖：
- 构建 state bank（多视图 embedding）
- 每步检索与经验注入文本格式
- `experience_top_n`、阈值配置的实际效果

### test_agent_integration.py — agent 循环内集成行为

```bash
python unit_test/state_level/test_agent_integration.py
```

覆盖：
- `search_experiences` 在 agent loop 中的调用时机
- `max_epoch`、state reset、日志落盘行为
- 工具 schema 中的注册与可调用性
