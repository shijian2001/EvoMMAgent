# Retrieval Pipeline 测试指南

## 一、vLLM 服务部署

### 1. Embedding 服务 (port 8001)

```bash
vllm serve Qwen/Qwen3-VL-Embedding-2B \
    --runner pooling \
    --port 8001 \
    --dtype float16
```

验证：
```bash
curl http://localhost:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-VL-Embedding-2B", "input": ["hello world"]}'
# 预期：返回 JSON，data[0].embedding 是一个 float 数组（维度 2048）
```

> 可选更大模型：`Qwen/Qwen3-VL-Embedding-8B`（维度 4096）

### 2. Reranker 服务 (port 8002)

Qwen3-VL-Reranker 需要 `hf_overrides` 和 chat template（已提供在 `template/qwen3_vl_reranker.jinja`）。

```bash
vllm serve Qwen/Qwen3-VL-Reranker-2B \
    --runner pooling \
    --port 8002 \
    --dtype float16 \
    --max-model-len 4096 \
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
    "model": "Qwen/Qwen3-VL-Reranker-2B",
    "query": "What color is the car?",
    "documents": ["A red car on the road", "Blue sky"]
  }'
# 预期：返回 results[].{index, relevance_score}，car 相关的 score 应更高
```

> 可选更大模型：`Qwen/Qwen3-VL-Reranker-8B`

---

## 二、运行测试

三个脚本，依赖从少到多。**从项目根目录运行。**

### test_local.py — 纯本地逻辑（无需任何服务）

```bash
python unit_test/test_local.py
```

| 测试项 | 逻辑 | 预期 |
|--------|------|------|
| Config 默认值 | 检查 `RetrievalConfig.enable=False`，`Config.retrieval` 集成 | 新功能默认关闭，不影响原有流程 |
| build_index_text | 用假 trace 构建索引文本 | 输出包含 question/tools/answer，非空 |
| MemoryBank 加载+搜索 | 写入随机 embedding → 加载 → cosine search | 返回 top-k 结果，含 retrieval_score 和完整 trace 数据 |
| MemoryBank 缺失处理 | 无 bank/ 目录时初始化 | 抛出 FileNotFoundError |

### test_services.py — Embedder + Reranker（需要 vLLM 服务）

```bash
python unit_test/test_services.py \
    --embedding_model Qwen/Qwen3-VL-Embedding-2B \
    --embedding_base_url http://localhost:8001/v1 \
    --rerank_model Qwen/Qwen3-VL-Reranker-2B \
    --rerank_base_url http://localhost:8002/v1
```

| 测试项 | 逻辑 | 预期 |
|--------|------|------|
| Embedder.encode_text | 编码 2 条文本 | 返回 shape=[2, D] 的向量，D>0 |
| Embedder.encode_batch | 4 条文本按 batch_size=2 分批编码 | 返回 shape=[4, D]，分两次 API 调用 |
| MemoryBank.build | 扫描假 trace → 过滤 correct → 批量编码 → 写入 bank/ | 生成 3 条记录（第 4 条 is_correct=False 被过滤），文件持久化 |
| 真实 embedding 搜索 | 用 "What is the color of the vehicle?" 查询 | 返回结果，car 相关 trace 的 score 应较高 |
| Reranker.rerank | 3 个候选，query="What color is the car?" | top-2 结果，car 相关候选 rerank_score 最高，分数降序 |

> Reranker 可选跳过：省略 `--rerank_model` 即可只测 Embedder。

### test_pipeline.py — 完整 Pipeline（需要全部服务 + LLM API）

```bash
python unit_test/test_pipeline.py \
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
| QueryRewriter 基础 | LLM 生成改写 query | 返回 ≥2 条 text_queries，第一条是原始 question |
| QueryRewriter 带 context | 传入上轮检索上下文 | LLM 基于上下文生成更有针对性的 query |
| Pipeline 单轮全功能 | rewrite → embed → search → rerank → summary | 返回非空 experience 字符串（2-3 句经验总结） |
| Pipeline 多轮 (2轮) | 第 1 轮检索后 sufficiency 判断 → 可能触发第 2 轮 | 返回非空 experience，日志中可见轮次信息 |
| Pipeline 关闭 rewrite | 跳过改写，直接用原始 question 检索 | 正常返回 experience |
| Pipeline 关闭 rerank | 跳过 reranker，按 retrieval_score 排序取 top-n | 正常返回 experience |

---

## 三、覆盖的代码模块

| 脚本 | 覆盖的新增文件 | 依赖 |
|------|---------------|------|
| `test_local.py` | `config.py (RetrievalConfig)`, `mm_memory/memory_bank.py` | 无 |
| `test_services.py` | `mm_memory/retrieval/embedder.py`, `mm_memory/retrieval/reranker.py`, `mm_memory/memory_bank.py (build)` | Embedding + Reranker vLLM |
| `test_pipeline.py` | `mm_memory/retrieval/query_rewriter.py`, `mm_memory/retrieval/pipeline.py` | 全部 |

---

## 四、排查

| 症状 | 检查 |
|------|------|
| test_local 失败 | import 路径错误，或 config.py / memory_bank.py 代码 bug |
| Embedder 连接失败 | `curl localhost:8001/v1/embeddings` 验证服务存活，model name 需一致 |
| Reranker 连接失败 | `curl localhost:8002/v1/rerank` 验证，注意需要 `hf_overrides` 和 chat template |
| QueryRewriter 返回空 | LLM API 连通性，或 JSONParser 解析失败（看 WARNING 日志） |
| Pipeline summary 为空 | `all_candidates` 为空 → 检索没命中，检查 embedding 维度是否匹配 |
