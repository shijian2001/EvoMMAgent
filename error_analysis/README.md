# 错误分析模块

通过外部大模型，分析 w/tool 轨迹中的错误 case，以及 direct 与 w/tool 之间的不一致 case。

## 配置

- **API Key**：通过 `--api_key` 参数传入，或设置环境变量 `ANALYSIS_API_KEY`
- **API Endpoint**：默认为 runway 端点，可通过 `--api_url` 覆盖；如需修改默认值，编辑 `client.py` 中的 `DEFAULT_API_URL`

## 使用方式

### 错误分析

分析 w/tool 中答错的 case，输出错误类别与根因分析：

```bash
python -m error_analysis error \
  --results /path/to/results.jsonl \
  --memory_dir /path/to/memory \
  --output_dir ./error_output \
  --api_key YOUR_API_KEY
```

### 对照分析

对比 direct（无工具）与 w/tool 在不一致 case 上的差异：

```bash
python -m error_analysis compare \
  --direct_results /path/to/direct/results.jsonl \
  --tool_results /path/to/tool/results.jsonl \
  --memory_dir /path/to/memory \
  --output_dir ./compare_output \
  --api_key YOUR_API_KEY
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--api_key` | `$ANALYSIS_API_KEY` | LLM API 密钥，也可通过环境变量传入 |
| `--api_url` | runway 端点 | 自定义 API 地址 |
| `--output_dir` | （必填） | 输出目录 |
| `--concurrency` | 10 | 最大并发 LLM 调用数 |

### 输出

每个命令产出：

- `*_details.jsonl` — 逐 case 的结构化分析结果
- `report.md` — 可读的汇总报告（统计表 + 子任务分布 + 逐 case 详情）

### 错误类别（多标签）

| 类别 | 说明 |
|------|------|
| `visual_perception` | Agent 或工具对图像内容识别有误 |
| `ineffective_tool_use` | 工具选择不当、调用冗余或目标错误 |
| `tool_misinterpretation` | 工具输出正确但 Agent 解读有误 |
| `reasoning_error` | 观察正确但推理链存在逻辑漏洞 |
| `instruction_following` | 未按要求格式作答或违反约束 |
| `no_answer` | 未能产出最终答案 |
