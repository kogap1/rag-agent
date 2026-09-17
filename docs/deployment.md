# 服务化部署与运维边界

## 1. 本地 REST 与 MCP

复制环境配置并设置 API Key：

```powershell
Copy-Item .env.example .env
# 编辑 .env：API_KEY 必须使用强随机值
uvicorn api:app --host 127.0.0.1 --port 8080
```

存活检查 `/v1/health/live` 不加载模型；就绪检查 `/v1/health/ready` 校验 SQLite 目录，并返回 LangGraph 编排方式、MCP 地址、LLM backend、鉴权开关和版本状态。首次真实查询仍可能触发 Embedding、Reranker 与本地 LLM 的延迟加载。

MCP Streamable HTTP 端点为 `/mcp/`，提供四个只读工具：`search_knowledge_base`、`query_knowledge_base`、`list_knowledge_documents`、`knowledge_base_health`。REST 可使用 `X-API-Key`，MCP 客户端建议使用 `Authorization: Bearer <API_KEY>`。

## 2. 对接 vLLM

本服务只依赖 OpenAI-compatible Chat Completions，不要求把 LoRA 或模型权重合并进本仓库。先单独启动并验证 vLLM，再配置：

```dotenv
LLM_BACKEND=openai_compatible
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
LLM_API_BASE=http://vllm-host:8000/v1
LLM_API_KEY=EMPTY
LLM_API_TIMEOUT_SECONDS=120
```

需要分别验证模型/聊天模板是否匹配、TTFT、P95、并发吞吐、错误率与 GPU 显存。未经压测的 GPU 利用率、并发数和量化参数不能写成项目结果。

## 3. Conda + CUDA 11.x 服务器

当前 LangGraph 1.x、MCP 2.x 和 LangChain 1.x 需要 Python 3.10+；官方 `torch+cu111` 停留在 PyTorch 1.10.1，不能与当前 Transformers 4.57 依赖闭环共存。因此，服务器显示 CUDA 11.1 时，请用 `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118` 让 `env` 阶段安装 PyTorch 2.6 官方 cu118 wheel，依靠 NVIDIA CUDA 11.x minor compatibility 运行，并在启动前实际分配 GPU 张量验证。

要求：Linux x86_64、Conda、`curl`、`nvidia-smi`，NVIDIA 驱动不低于 450.80.02。

已有一个装好依赖的环境时，只起服务，不建环境也不装包：

```bash
cp .env.example .env
conda activate <已有且已装好依赖的环境>
PUBLIC_HOST=rag.example.com bash scripts/server.sh serve
```

`serve` 阶段会自动生成并保存 API Key，复用当前已激活的非 `base` Conda 环境；也可用 `CONDA_ENV_NAME=<已有环境名>` 显式指定。环境不存在时直接停止并说明原因，不会静默回退到裸 `python`。启动前会做离线依赖导入和 CUDA 检查。`PUBLIC_HOST` 会加入 MCP host allowlist；如果经 HTTPS 反向代理访问，还需在 `.env` 的 `MCP_ALLOWED_ORIGINS` 中加入实际来源，例如 `https://rag.example.com`。

需要连环境一起准备时（`env` 阶段会创建环境并安装 torch 与 `requirements.txt`）：

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118 bash scripts/server.sh env
```

```bash
tail -f runs/service.log
bash scripts/server.sh stop
```

无 GPU 的测试机启动时可显式设置 `REQUIRE_CUDA=0`，生产启动默认禁止静默回退 CPU。分阶段流程与排障见 [服务器部署流程命令](server-deploy-runbook.md)。

## 4. Docker（可选）

```bash
bash scripts/server.sh docker
docker compose logs -f agentic-rag
```

镜像使用非 root 用户，数据、向量、SQLite、运行日志和模型目录均挂载到宿主机。默认只启动一个 Uvicorn worker；本地模型与多 worker 会重复占用内存/显存。Compose 中的内存上限只是保护栏，需要按模型与压测结果调整。

`API_MAX_CONCURRENCY` 默认是 1：这是本地 Transformers/Embedding/Reranker 共用模型实例时的保守值。REST 与 MCP 查询共享该背压边界。OpenAI-compatible 生成后端虽然可以并发 HTTP 请求，但检索模型仍在本进程共享；只有在线程安全和容量压测通过后才应提高。

## 5. 监控建议

`/metrics` 输出 Prometheus 文本格式，至少应监控：

- HTTP 2xx/4xx/5xx 数量和在途请求；
- 累计耗时派生的平均延迟，以及外部网关记录的 P95/P99；
- Agent JSONL 中的 Grounding、工具失败、错误类型与 token；
- SQLite `building_versions` 长时间非零或 `failed_versions` 增长；
- vLLM 的 TTFT、队列等待、KV cache、吞吐和 GPU 显存。

## 6. 升级旧索引

旧版 Chroma 分块没有 `version_id`。兼容模式仍可读取，但不会获得新版本目录的一致性保证。保留原目录备份后，用原始 PDF 重建：

```powershell
python create_db.py data --replace
```

重建完成后检查 `/v1/documents` 的 `status=active`、`chunk_count>0`，再运行固定评测集。不要只以“服务能启动”作为升级验收。
