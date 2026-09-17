# 评测集说明

每行是一个 JSON 对象，字段：

```json
{
  "id": "rule-01",
  "question": "问题",
  "expected_keywords": ["期望出现的关键词"],
  "expected_sources": ["期望被检索到的文档名片段"],
  "require_grounding": true,
  "forbidden_keywords": ["已知错误表述"],
  "history": [{"role": "user", "content": "前一轮用户问题"}],
  "expect_no_answer": false,
  "note": "这条用例测什么"
}
```

- `history`：多轮/指代任务的前置对话，传给 Agent 的上下文管理器。留空表示单轮。
- `expect_no_answer`：文档确实未涉及、应拒答（不引用证据）的任务。判定为成功要求：`grounded=False`、回答非空、无运行错误、无禁止词。用于把“不会就说不”固化为可评测行为。
- `note`：仅用于说明用例意图，不影响判定。

任务成功（单轮普通用例）要求：所有期望关键词出现在答案中、所有期望来源被检索、禁止词不出现、引用校验通过且没有运行错误。禁止词用于把人工发现的幻觉固化为回归测试。`require_grounding=false` 仅适用于文档清单等不需要页码证据的任务。

## 任务类型覆盖

当前 `dataset.example.jsonl` 覆盖：直接事实、数值抽取、表格读取、多跳归纳、无答案/拒答、多轮指代、查询改写、Prompt Injection 和文档清单。新增用例时建议标注 `note` 并人工复核问题、标准与来源，尤其要复核：

- **无答案用例**：模型可能检索到话题相近的证据后强行作答，此时 `grounded=True` 但并未回答问题，当前关键词+来源判定会误判为成功。这类用例需要人工核验或引入更严格的 Judge。
- **查询改写用例**：口语化问题是否真的触发无召回→改写路径，取决于检索质量，需跑通后确认。
- **注入用例**：仅验证“被诱导不引用时判失败”，不声称覆盖全部注入手段。

## 指标

报告 `reports/*.md` 除正确率类指标外，还包含：

- **检索质量**：根据 `expected_sources` 计算 Recall@K（期望来源被 Top-K 命中的比例）与 MRR（第一个相关来源排名的倒数均值）。没有期望来源的文档清单/拒答用例不进入这两个指标的分母。
- **检索命中率（Retrieval Hit Rate）**：期望来源至少命中一个的任务占比，是需求口径中「检索命中」的直接对应指标。
- **引用准确性（Citation Accuracy）**：回答中的引用编号能对应到本轮证据的比例；无引用的回答与拒答用例不计入分母。它校验的是「编号合法且指向本轮证据」，不等价于自然语言蕴含。
- **Token/Cost**：`mean_prompt_tokens`、`mean_completion_tokens`、`mean_total_tokens`、`tokens_per_success`（每个成功任务的平均 token）。在 `.env` 配置 API 单价后，额外给出 `mean_cost_usd` 和 `cost_per_success_usd`；本地模型成本为 0。
- **上下文管理**：`mean_context_chars`（压缩后）、`mean_context_raw_chars`（压缩前）、`context_compression_ratio`（节省比例）。单轮任务 history 为空时该比例接近 0，需用多轮任务评测才能真正体现压缩收益。
- 原有：Task Success、Grounding、Tool Success、Recovery、Policy Violation、Stability@N、步骤数、平均/P95 时延。

## 复现实验矩阵

Agent 与 Baseline 必须使用同一份数据、模型、知识库、参数和硬件，建议每题重复 3 次报告 Stability@3。推荐一次跑完消融矩阵：

```powershell
python run_ablation.py --repeats 3 --dataset eval/dataset.example.jsonl
```

等效的四条命令（可用于单条重跑）：

```powershell
python evaluate.py --mode baseline --repeats 3
python evaluate.py --mode baseline --repeats 3 --disable-reranker
python evaluate.py --mode agent --repeats 3
python evaluate.py --mode agent --repeats 3 --disable-reranker
```

对比任意两份报告：

```powershell
python compare_reports.py reports/baseline-*.json reports/agent-*.json --out reports/compare.md
```

报告的实验元数据（模型、embedding、reranker、设备、重复次数、计费单价）会一并写入 JSON，避免脱离实验条件引用百分比。
