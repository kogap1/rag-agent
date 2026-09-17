# 架构、治理与恢复设计

## 信任边界

用户问题和 PDF 文本均视为不可信输入。模型只负责从固定枚举中提出工具计划；真正执行前由 `ToolRegistry` 再次验证工具名和参数。Agent 主链路只注册只读工具，上传、覆盖和清空知识库必须由用户在 Streamlit UI 主动点击。

工具结果在进入 Prompt 前按字符预算截断。系统提示明确要求忽略文档内的越权指令，从结构约束和提示约束两层降低间接 Prompt Injection 风险。该设计不声称彻底解决注入攻击。

HTTP 边界可配置单一 API Key，REST 支持 `X-API-Key`，REST/MCP 均支持 `Authorization: Bearer`。上传文件同时校验扩展名、大小和 PDF magic bytes，服务仅将文件写入配置的上传目录。API Key 只是服务级门禁，不是用户身份或文档级 ACL。

## 文档版本与一致性

- 原始文件按 SHA-256 存入内容寻址对象目录，同一内容重复上传不会产生多份文件。
- SQLite 目录保存 stable document/version ID 与 `building/active/inactive/failed` 状态；Chroma 保存带 version ID 的分块向量。
- 新版本所有分块写入完成后，才在 SQLite 事务内从 `building` 切换到 `active`。查询仅过滤 active version，避免半成品可见。
- 激活后保留最近 `KEEP_HISTORY_VERSIONS` 个已激活版本的向量，供升级失败后回退；只有超出回退窗口的版本才回收向量并被标记为 `pruned`，回收失败不影响读路径。
- 写入失败会删除本次已写入的分块并记录 `failed` 状态，原 active version 保持不变；`POST /v1/documents/rollback` 可在单个事务内把 active 切回上一个健康版本。
- `mark_pruned` 会把分块数归零，使被裁剪的版本无法被误回退；因此回退能力与回退窗口是同一条约束。
- 同一逻辑文档的并发写入会被拒绝；进程崩溃遗留的 `building` 状态超过 `INGESTION_LEASE_SECONDS` 后可由重试接管。
- 该方案解决单实例中的读写切换，不等价于跨节点两阶段提交。多副本写入需改用集中式任务队列和支持事务/别名切换的向量服务。

## 状态与终止

- LangGraph 节点：`prepare_context` → `plan_intent` → `execute_tool`，再按结果条件路由到文档清单、查询改写、拒答或答案生成；生成后进入引用校验，最多执行一次引用修复。
- 对话状态：最近 `MAX_HISTORY_MESSAGES` 条原文 + 更早用户问题主题。
- 字符预算：最终上下文不超过 `MAX_CONTEXT_CHARS`；每次运行记录压缩前（raw）与压缩后字符数，用于计算压缩率。
- 模型用量：每次运行从 0 累计 prompt/completion token，随任务结果写入日志与评测报告。
- 工具状态：名称、成功状态、尝试次数、错误类型和延迟。
- 可追溯性：每次运行生成 `run_id`，每个 `AgentStep` 携带 `record_id` 与 `evidence`（检索步骤给出 chunk_id，校验步骤给出版本化的引用编号），随运行日志落盘，使每一步都能回溯到具体记录。
- 任务终止：列表工具返回、获得证据并完成引用校验、无证据（拒答）、工具不可恢复失败或达到 Agent 步骤上限。

## 错误策略

| 错误 | 策略 | 原因 |
|---|---|---|
| LLM 规划输出非法 JSON | 降级为原问题检索 | 不让规划器成为单点故障 |
| 未注册/非法参数 | 立即拒绝 | 重试无法修复，且可能越权 |
| 工具超时/向量库异常 | 有限重试 | 只读、幂等调用重试风险较低 |
| 首轮无召回 | 改写查询后重试，轮数上限由 `MAX_RETRIEVAL_ROUNDS` 决定 | 控制步骤与时延预算 |
| 达到检索轮数上限仍无证据 | 明确拒答 | 避免无依据作答和无效循环 |
| 引用编号非法 | 标记未通过并提示核验 | 不把格式不可信的答案伪装成完成 |
| 日志写入失败 | 不阻断回答 | Telemetry 不是主任务完成条件 |

## 幂等性

只读工具以“工具名 + 规范化参数”的 SHA-256 作为单次 Agent 实例内缓存键，重复调用直接复用结果。知识库以“逻辑文档 ID + 文件 SHA-256”生成稳定版本 ID，以“版本 ID + 分块序号 + 分块 SHA-256”生成稳定 chunk ID；重试不会制造新的逻辑版本。

## 服务进程边界

- FastAPI 将同步模型执行放入工作线程，使用非阻塞容量门限制同时进入模型的请求；超限立即返回 429。
- MCP Streamable HTTP 挂载在 `/mcp/`，暴露 `search_knowledge_base`（纯检索）、`query_knowledge_base`（检索+生成）、`list_knowledge_documents` 与 `knowledge_base_health` 四个只读工具；上传、清空、版本切换与回退不通过 MCP 开放。
- MCP 与 REST 复用同一服务实例和 API Key 边界，并通过 host/origin allowlist 降低 DNS rebinding 风险。
- 每个 HTTP 请求生成或透传 `X-Request-ID`，并输出请求量、状态码、累计耗时、在途请求和异常计数。
- 生成模型可在进程内用 Transformers 加载，也可通过 OpenAI-compatible HTTP 接口拆到 vLLM；Embedding/Reranker 仍由本服务加载。
- Docker 镜像使用非 root 用户并默认单 worker，避免每个 worker 重复占用模型显存。该限制应通过压测后再调整。
