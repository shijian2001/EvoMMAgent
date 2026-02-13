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

验证：
```bash
curl http://localhost:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3vl-embed", "input": ["hello world"]}'
# 预期：返回 JSON，data[0].embedding 是一个 float 数组（维度 4096）
```

### 2. Reranker 服务 (port 8002)

Qwen3-VL-Reranker 需要 `hf_overrides` 和 chat template（已提供在 `template/qwen3_vl_reranker.jinja`）。

```bash
vllm serve /Path/to/Qwen3-VL-Reranker-2B \
    --runner pooling \
    --port 8002 \
    --dtype float16 \
    --max-model-len 32678 \
    --served-model-name qwen3vl-reranker \
    --hf-overrides '{
      "architectures": ["Qwen3VLForSequenceClassification"],
      "classifier_from_token": ["no", "yes"],
      "is_original_qwen3_reranker": true
    }' \
    --chat-template template/qwen3_vl_reranker.jinja
```

验证：
```bash
curl http://localhost:8002/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3vl-reranker",
    "query": "What color is the car?",
    "documents": ["A red car on the road", "Blue sky"]
  }'
# 预期：返回 results[].{index, relevance_score}，car 相关的 score 应更高
```

> 可选更大模型：`Qwen/Qwen3-VL-Reranker-8B`

---

## 二、测试目录结构

```
unit_test/
├── helpers.py              # 共享 fixtures / 工具函数
├── test_answer_match.py    # 答案匹配测试（与 retrieval 无关）
├── README.md               # 本文件
├── trace_level/            # Trace-level (任务级) 检索测试
│   ├── test_local.py       # 纯本地逻辑
│   ├── test_services.py    # Embedder + Reranker 服务
│   └── test_pipeline.py    # 完整 TracePipeline (需 LLM API)
└── state_level/            # State-level (状态级) 检索测试
    ├── test_local.py       # StateBank、state_to_text、config mode
    └── test_pipeline.py    # StatePipeline end-to-end (需 Embedder)
```

---

## 三、Trace-Level 测试

### test_local.py — 纯本地逻辑（无需任何服务）

```bash
python unit_test/trace_level/test_local.py
```

| 测试项 | 逻辑 | 预期 |
|--------|------|------|
| Config 默认值 | 检查 `RetrievalConfig.enable=False`，`Config.retrieval` 集成 | 新功能默认关闭，不影响原有流程 |
| build_index_text | 用假 trace 构建索引文本 | 输出包含 question/tools/answer，非空 |
| MemoryBank 加载+搜索 | 写入随机 embedding → 加载 → cosine search | 返回 top-k 结果，含 retrieval_score 和完整 trace 数据 |
| MemoryBank 缺失处理 | 无 bank/ 目录时初始化 | 抛出 FileNotFoundError |

### test_services.py — Embedder + Reranker（需要 vLLM 服务）

```bash
python unit_test/trace_level/test_services.py \
    --embedding_model Qwen/Qwen3-VL-Embedding-2B \
    --embedding_base_url http://localhost:8001/v1 \
    --rerank_model Qwen/Qwen3-VL-Reranker-2B \
    --rerank_base_url http://localhost:8002/v1
```

| 测试项 | 逻辑 | 预期 |
|--------|------|------|
| Embedder.encode_text | 编码 2 条文本 | 返回 shape=[2, D] 的向量，D>0 |
| Embedder.encode_batch | 4 条文本按 batch_size=2 分批编码 | 返回 shape=[4, D]，分两次 API 调用 |
| MemoryBank.build | 扫描假 trace → 过滤 correct → 批量编码 → 写入 bank/ | 生成 3 条记录，文件持久化 |
| 真实 embedding 搜索 | 用 "What is the color of the vehicle?" 查询 | 返回结果，car 相关 trace 的 score 应较高 |
| Reranker.rerank | 3 个候选，query="What color is the car?" | top-2 结果，car 相关候选 rerank_score 最高 |

### test_pipeline.py — 完整 TracePipeline（需要全部服务 + LLM API）

```bash
python unit_test/trace_level/test_pipeline.py \
    --embedding_model Qwen/Qwen3-VL-Embedding-2B \
    --embedding_base_url http://localhost:8001/v1 \
    --rerank_model Qwen/Qwen3-VL-Reranker-2B \
    --rerank_base_url http://localhost:8002/v1 \
    --llm_model qwen3-vl-8b-instruct \
    --llm_base_url https://maas.devops.xiaohongshu.com/v1 \
    --llm_api_key YOUR_API_KEY
```

| 测试项 | 逻辑 | 预期 |
|--------|------|------|
| QueryRewriter 基础 | LLM 生成改写 query | 返回 ≥2 条 text_queries |
| Pipeline 单轮全功能 | rewrite → embed → search → rerank → summary | 返回非空 experience |
| Pipeline 多轮 (2轮) | 第 1 轮检索后 sufficiency 判断 → 可能触发第 2 轮 | 返回非空 experience |
| Pipeline 关闭 rewrite/rerank | 分别跳过改写或 reranker | 正常返回 experience |

---

## 四、State-Level 测试

### test_local.py — 纯本地逻辑（无需任何服务）

```bash
python unit_test/state_level/test_local.py
```

| 测试项 | 逻辑 | 预期 |
|--------|------|------|
| Config mode 默认值 | `RetrievalConfig.mode="state"`, `min_q_value=5` | 新 state mode 字段正确 |
| convert_trace_to_trajectory | 原始 trace → MDP trajectory | think+action 合并为 atomic action |
| StateBank.state_to_text | 序列化 s_0, s_1, s_2 | s_0 只有 query，s_t 包含前 t 个 action 摘要 |
| StateBank 加载+搜索 | 合成 embedding → cosine search + Q-value 过滤 | 返回 top-k，Q < min_q 被过滤 |
| StateBank 缺失处理 | 无 state_bank/ 目录 | 抛出 FileNotFoundError |

### test_pipeline.py — StatePipeline 端到端（需要 Embedder 服务）

```bash
python unit_test/state_level/test_pipeline.py \
    --embedding_model Qwen/Qwen3-VL-Embedding-2B \
    --embedding_base_url http://localhost:8001/v1
```

| 测试项 | 逻辑 | 预期 |
|--------|------|------|
| Build state bank | 用真实 embedding 构建 state bank | 生成正确数量的 state，文件持久化 |
| StatePipeline.retrieve (top-1) | embed → cosine search → 取 top-1 experience | 返回单行非空 experience |
| StatePipeline.retrieve (top-3) | experience_top_n=3 | 返回 ≤3 行 experience |
| 极端阈值 | min_q=10, min_score=0.99 | 返回空字符串 |

---

## 五、排查

| 症状 | 检查 |
|------|------|
| test_local 失败 | import 路径错误，或 config.py / memory_bank.py / state_bank.py 代码 bug |
| Embedder 连接失败 | `curl localhost:8001/v1/embeddings` 验证服务存活，model name 需一致 |
| Reranker 连接失败 | `curl localhost:8002/v1/rerank` 验证，注意需要 `hf_overrides` 和 chat template |
| QueryRewriter 返回空 | LLM API 连通性，或 JSONParser 解析失败（看 WARNING 日志） |
| Pipeline summary 为空 | `all_candidates` 为空 → 检索没命中，检查 embedding 维度是否匹配 |
| State retrieval 为空 | Q-value 阈值过高，或 state_bank/ 未构建 (运行 `scripts/build_state_bank.py`) |
