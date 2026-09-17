# 服务器部署流程命令（NVIDIA A40）

面向多卡 GPU 服务器（如 4×A40）的**分阶段可复制命令手册**。环境选型依据见 [A40 Conda 环境配置](conda-a40-setup.md)，服务化边界与监控口径见 [服务化部署说明](deployment.md)。

## 0. 目标机器的现状与结论

`nvidia-smi` 实测：

| 项目 | 实测值 | 对部署的影响 |
|---|---|---|
| GPU | 4× NVIDIA A40，单卡 46068 MiB | 单卡即可跑 7B（bf16 约 15GB），余量充足 |
| 驱动 | 535.183.01 | 对应 CUDA 12.2；按 CUDA 12.x minor version compatibility，可直接用 cu124 轮子，**无需升级驱动** |
| CUDA Version | 12.2 | 见上 |
| 计算能力 | sm_86（Ampere） | cu124 官方轮子已含 sm_86；支持 bf16，代码自动走 `torch.bfloat16` |
| 当前占用 | 4 张卡各 13MiB / 0% | 全部空闲，仅 Xorg 占用 4MiB；**没有互斥任务，可以随时部署** |
| 显存约束 | — | **必须设 `CUDA_VISIBLE_DEVICES=0`**：`agentic_rag/models.py` 用 `device_map="auto"`，不设会把模型切到 4 张卡上 |

结论：驱动够新、卡够空，走 **Conda + torch 2.6.0 + cu124 + Python 3.10**，单卡固定，生成模型用 `qwen/Qwen2.5-7B-Instruct`。

---

## 1. 前置检查

```bash
# 要有：conda、curl、nvidia-smi
conda --version && curl --version | head -n1 && nvidia-smi --query-gpu=index,name,memory.total --format=csv

# 确认 4 张卡都没被别的任务占住（Processes 段应只有 Xorg）
nvidia-smi

# 磁盘：模型缓存约需 20GB（BGE-M3 2.2G + Reranker 1.1G + 7B bf16 15G）
df -h /data

# 系统依赖（pypdf/chromadb 通常不需要额外系统库，以下用于排障）
sudo apt-get update && sudo apt-get install -y curl git rsync
```

---

## 2. 把代码放到服务器

> 下文统一以 `/data/apps/rag-agent-upgrade` 作为部署路径示例，按实际路径替换即可。

```bash
# 方式 A：本机推（在本地项目父目录执行）
rsync -avz --exclude .backup --exclude __pycache__ --exclude runs --exclude reports \
  ./rag-agent-upgrade/ user@your-server:/data/apps/rag-agent-upgrade/

# 方式 B：服务器上拉
git clone <你的仓库地址> /data/apps/rag-agent-upgrade

# 进入项目并给脚本执行权限
cd /data/apps/rag-agent-upgrade
chmod +x create_conda_env.sh run_server.sh scripts/*.sh
```

知识库源 PDF 用 `scp` 放到 `data/`：

```bash
scp ./your-rules.pdf user@your-server:/data/apps/rag-agent-upgrade/data/
```

---

## 3. 建 Conda 环境（完整命令行）

复用项目自带的 `create_conda_env.sh`，它会建环境 → 装 PyTorch → 装 `requirements.txt` → `pip check` → 导入与 CUDA 实算自检。

```bash
cd /data/apps/rag-agent-upgrade

CONDA_ENV_NAME=rag-agent \
PYTHON_VERSION=3.10 \
TORCH_VERSION=2.6.0 \
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
REQUIRE_CUDA=1 \
INSTALL_DEV=1 \
./create_conda_env.sh
```

期望输出（关键三行）：

```text
检测到 NVIDIA 驱动: 535.183.01
Python/PyTorch 环境就绪: torch=2.6.0+cu124, runtime_cuda=12.4
cuda_available=True
gpu=NVIDIA A40, allocation_check=2.0
```

等价的**手工逐条命令**（需要自己控制每一步时用）：

```bash
conda create -y -n rag-agent python=3.10 pip
conda activate rag-agent
python -m pip install --upgrade pip setuptools wheel

# 务必要带 --index-url，否则装到 CPU 轮子
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt   # 仅测试/开发需要

python -m pip check
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
```

预计算：整个环节首次执行需要拉取约 2.5GB 的 CUDA 轮子。

---

## 4. 配置 `.env`

```bash
cd /data/apps/rag-agent-upgrade
cp .env.example .env
chmod 600 .env
```

改这几项（其余保持默认）：

```bash
# 生成强随机 API Key 并写入
sed -i "s|^API_KEY=.*|API_KEY=$(openssl rand -hex 32)|" .env

# 4 卡机器固定单卡，避免 device_map="auto" 把模型摊到 4 张卡
sed -i "s|^CUDA_VISIBLE_DEVICES=.*|CUDA_VISIBLE_DEVICES=0|" .env

# A40 上用 7B；注意基线与 Agent 评测必须用同一模型
sed -i "s|^LLM_MODEL=.*|LLM_MODEL=qwen/Qwen2.5-7B-Instruct|" .env

# 显存充裕，可放开并发；做评测时改回 1 以保持可比
sed -i "s|^API_MAX_CONCURRENCY=.*|API_MAX_CONCURRENCY=2|" .env

# 复用环境名与 CUDA 强校验
sed -i "s|^CONDA_ENV_NAME=.*|CONDA_ENV_NAME=rag-agent|" .env
sed -i "s|^REQUIRE_CUDA=.*|REQUIRE_CUDA=1|" .env

# 经域名/IP 访问 MCP 时必须加进 host 白名单
sed -i "s|^MCP_ALLOWED_HOSTS=.*|MCP_ALLOWED_HOSTS=127.0.0.1:*,localhost:*,[::1]:*,rag.example.com,rag.example.com:*|" .env

grep -E '^(CUDA_VISIBLE_DEVICES|LLM_MODEL|API_MAX_CONCURRENCY|CONDA_ENV_NAME|REQUIRE_CUDA|MCP_ALLOWED_HOSTS)=' .env
```

| 变量 | 服务器取值 | 原因 |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | `0` | 不设时 `device_map="auto"` 会跨卡切分；固定单卡便于隔离排障 |
| `DEVICE` | `auto`（默认） | 自动选 cuda；bf16 由代码按 `is_bf16_supported()` 自动判定 |
| `LLM_MODEL` | `qwen/Qwen2.5-7B-Instruct` | bf16 约 15GB，单卡 46GB 余量充足 |
| `API_MAX_CONCURRENCY` | `2` | 本地 Transformers/Embedding/Reranker 共用模型实例，别一次放太开 |
| `REQUIRE_CUDA` | `1` | `run_server.sh` 启动前做 CUDA 检查，禁止静默回退 CPU |

> 改 `LLM_MODEL` **不需要重建向量库**：向量库只依赖 Embedding 与 Reranker。

---

## 5. 环境校验（离线，不下载模型）

```bash
cd /data/apps/rag-agent-upgrade
CONDA_ENV_NAME=rag-agent bash scripts/verify.sh
```

会依次跑：语法检查 → 全量 pytest → 离线端到端冒烟（不下载模型）。

期望：`compileall 通过`、`pytest 44 passed`、`全部验证通过。`

---

## 6. 构建向量知识库（首次会下载模型）

```bash
cd /data/apps/rag-agent-upgrade
conda activate rag-agent
python create_db.py data --replace
```

首次会从 ModelScope 下载 BGE-M3、BGE Reranker 与 7B 生成模型（合计约 20GB，视带宽十几分钟到半小时）。

另开一个终端观察下载与显存：

```bash
watch -n2 nvidia-smi
```

建库完成后核对状态（`status=active` 且 `chunk_count>0` 才算成功）：

```bash
python - <<'PY'
from agentic_rag import Settings
from agentic_rag.knowledge_base import KnowledgeBase
from agentic_rag.models import ModelRuntime

settings = Settings.from_env()
for r in KnowledgeBase(settings, ModelRuntime(settings)).document_records():
    print(f"{r.source} | {r.status} | {r.chunk_count} 块 | version={(r.active_version_id or '-')[:12]}")
PY
```

---

## 7. 启动服务

```bash
cd /data/apps/rag-agent-upgrade
CONDA_ENV_NAME=rag-agent REQUIRE_CUDA=1 PUBLIC_HOST=rag.example.com bash run_server.sh
```

脚本会做离线依赖导入 + CUDA 检查，不通过直接停止并列出原因；通过后后台启动 Uvicorn（`0.0.0.0:8080`，单 worker），并轮询存活与就绪探针。

期望输出：

```text
Agentic RAG 已启动并通过健康检查
REST API: http://<server>:8080/docs
MCP:      http://<server>:8080/mcp/
Metrics:  http://<server>:8080/metrics
进程PID:   <pid>
查看日志: tail -f runs/uvicorn.log
停止服务: kill <pid>
```

> 注意：`/v1/health/live` 不加载模型；**首次真实问答**才会懒加载 Embedding/Reranker/LLM，可能等待数十秒，不是卡死。

---

## 8. 在线验收

```bash
cd /data/apps/rag-agent-upgrade
KEY=$(sed -n 's/^API_KEY=//p' .env | tail -n1)
BASE=http://127.0.0.1:8080

# 1) 存活探针
curl -s -o /dev/null -w 'live=%{http_code}\n' $BASE/v1/health/live

# 2) 就绪探针
curl -s -H "X-API-Key: $KEY" $BASE/v1/health/ready | head -c 400; echo

# 3) 文档清单
curl -s -H "X-API-Key: $KEY" $BASE/v1/documents; echo

# 4) 真实问答
curl -s -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"question":"综合测评如何计算？","history":[]}' $BASE/v1/query | head -c 800; echo

# 5) Prometheus 指标
curl -s -o /dev/null -w 'metrics=%{http_code}\n' -H "X-API-Key: $KEY" $BASE/metrics

# 6) 鉴权边界：无 Key 必须被拒绝
curl -s -o /dev/null -w 'no_key=%{http_code}\n' -X POST -H 'Content-Type: application/json' \
  -d '{"question":"hi","history":[]}' $BASE/v1/query
```

判定标准：`live=200`、`ready=200`、`metrics=200`、`no_key=401`，且第 4 步返回的 `grounded=true` 并带 `来源，第 N 页` 引用。

---

## 9. 常驻化（systemd，开机自启）

`nohup` 起的进程重启机器就没了。生产上用 systemd：

```bash
# 生成单元文件（自动填好绝对路径与解释器）
cd /data/apps/rag-agent-upgrade
bash scripts/deploy_server.sh systemd

# 安装并启用
sudo cp deploy/rag-agent.service /etc/systemd/system/rag-agent.service
sudo systemctl daemon-reload
sudo systemctl enable --now rag-agent
systemctl status rag-agent --no-pager
```

先停掉手工启动的进程，避免争抢 8080 端口：

```bash
bash scripts/deploy_server.sh stop
```

日志与重启：

```bash
journalctl -u rag-agent -f
sudo systemctl restart rag-agent
```

---

## 10. 反向代理与防火墙

对外只暴露 443，8080 只在回环监听更稳妥。

```bash
sudo ufw allow 443/tcp
sudo ufw status
# 若直连 8080，才需要：sudo ufw allow 8080/tcp
```

nginx 关键配置（`client_max_body_size` 必须 ≥ `MAX_UPLOAD_BYTES` 即 50MiB）：

```nginx
server {
    listen 443 ssl;
    server_name rag.example.com;

    client_max_body_size 50m;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;   # 首次问答要懒加载模型
        proxy_buffering off;       # /mcp/ 走 Streamable HTTP，关闭缓冲
    }
}
```

经 HTTPS 代理后，浏览器侧 MCP 客户端还需把来源加进白名单：

```bash
sed -i "s|^MCP_ALLOWED_ORIGINS=.*|MCP_ALLOWED_ORIGINS=http://127.0.0.1:*,http://localhost:*,http://[::1]:*,https://rag.example.com|" .env
```

---

## 11. 可重复执行的一键脚本

上面的流程已固化为 `scripts/deploy_server.sh`，幂等，可反复执行：

```bash
cd /data/apps/rag-agent-upgrade

# 全流程：环境 → 配置 → 校验 → 建库 → 起服务 → 验收
PUBLIC_HOST=rag.example.com bash scripts/deploy_server.sh all

# 或分阶段（每步可单独重跑）
bash scripts/deploy_server.sh env        # 系统检查 + 建/复用 Conda 环境
bash scripts/deploy_server.sh config     # 写 .env（A40 推荐参数 + API Key + MCP 白名单）
bash scripts/deploy_server.sh verify     # 语法 + pytest + 离线冒烟
bash scripts/deploy_server.sh build      # 下载模型 + 建向量库
bash scripts/deploy_server.sh serve      # 启动服务并通过健康检查
bash scripts/deploy_server.sh accept     # 在线接口验收
bash scripts/deploy_server.sh systemd    # 生成 systemd 单元

bash scripts/deploy_server.sh status     # 状态总览（服务 / 环境 / 知识库）
bash scripts/deploy_server.sh logs       # tail -f 服务日志
bash scripts/deploy_server.sh stop       # 停止服务
```

环境已就绪、只想重跑后面几步时：

```bash
SKIP_ENV_CREATE=1 SKIP_BUILD=1 bash scripts/deploy_server.sh all
```

### 关于沿用已有的 Conda 环境

`CONDA_ENV_NAME` 默认是 `rag-agent`。如果服务器上已经有装好依赖的环境（例如 `rag_a40`），显式指定即可复用，省掉一次约 2.5GB 的 CUDA 轮子下载：

```bash
CONDA_ENV_NAME=rag_a40 PUBLIC_HOST=rag.example.com bash scripts/deploy_server.sh all
```

脚本会先做一次**只读**依赖探测：依赖齐全就直接复用；不齐才调用 `create_conda_env.sh` 补齐（该脚本复用已存在环境，不会重建）。**注意补齐动作会往该环境里装 torch / requirements**——若 `rag_a40` 是给别的项目用的，请不要复用，保持默认让它新建 `rag-agent`。

想先确认那个环境里到底有什么（不依赖 `conda run`，老版本 conda 也能跑）：

```bash
"$(conda env list | awk '$1=="rag_a40"{print $NF}')/bin/python" -c \
  "import torch, langgraph, mcp, fastapi, chromadb, sentence_transformers, streamlit; print(torch.__version__, torch.cuda.is_available())"
```

### 中断了怎么定位

脚本装了 `ERR` trap，任何真实失败都会打印**行号、失败命令、退出码**：

```text
   ❌ 部署中断：第 199 行执行失败（退出码 1）
      失败命令：nvidia-smi ...
      文件位置：scripts/deploy_server.sh
      查看该行：sed -n 199p scripts/deploy_server.sh
```

看到这段直接按行号去看那一行即可，不会再出现「跑到一半静默回到提示符」。若在更早拷过去的副本上遇到静默中断，用跟踪模式拿真实位置：

```bash
bash -x scripts/deploy_server.sh env 2>&1 | tail -30
```

---

## 12. 运维速查

```bash
# 服务状态与进程
bash scripts/deploy_server.sh status
kill -0 "$(cat runs/uvicorn.pid)" && echo running

# 日志
tail -f runs/uvicorn.log            # 手工启动
journalctl -u rag-agent -f          # systemd 启动
tail -f runs/agent_runs.jsonl       # Agent 轨迹（run_id / 每步 record_id / evidence）

# 资源
nvidia-smi                          # 显存与利用率
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv -l 2

# 评测（生成 reports/*.json 与 *.md）
bash scripts/run_local.sh eval
python run_ablation.py --repeats 3 --dataset eval/dataset.example.jsonl
```

---

## 13. 排障表

| 现象 | 原因与处理 |
|---|---|
| `torch.cuda.is_available()=False` | 装到了 CPU 轮子。重装：`python -m pip uninstall -y torch && python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124` |
| 显存被摊到 4 张卡 | `.env` 未设 `CUDA_VISIBLE_DEVICES=0`。改完后必须**重启进程**（CUDA 在首次初始化时读取该变量） |
| `CUDA error: no kernel image is available` | 轮子与架构不匹配。确认装的是 cu124 官方轮子而非 `+cpu` 版本 |
| `run_server.sh` 报「未指定可复用的 Conda 环境」 | 未激活环境且未传环境名。用 `CONDA_ENV_NAME=rag-agent bash run_server.sh` |
| 启动报缺少模块 | 环境没装全依赖。`"$(conda env list \| awk '$1=="rag-agent"{print $NF}')/bin/python" -m pip install -r requirements.txt` |
| `conda: error: unrecognized arguments: --no-capture-output` | 旧版 conda 不认这个参数。脚本已改为**直接调用环境自带解释器**，不再使用 `conda run`；拉取最新 `create_conda_env.sh` / `run_server.sh` 即可 |
| `/v1/health/ready` 返回非 200 | 看 `runs/uvicorn.log`；就绪探针会校验 SQLite 目录，确认 `state/` 可写 |
| 首次问答很慢或超时 | Embedding/Reranker/7B 懒加载。nginx `proxy_read_timeout` 调到 300s |
| 下载模型慢 | 换 `MODEL_CACHE_DIR` 到大盘；或先手工 `modelscope download` 到对应目录 |
| `/mcp/` 返回 400/403 | 访问用的 host 不在 `MCP_ALLOWED_HOSTS`。按第 4 节加入域名后重启 |
| 端口被占 | `bash scripts/deploy_server.sh stop`；或查 `ss -lntp \| grep 8080` |
| 上传报 413 | nginx `client_max_body_size` 小于 `MAX_UPLOAD_BYTES`（50MiB） |
| 脚本「跑到一半静默回到提示符」 | 旧副本才会有。当前脚本装了 `ERR` trap，会打印行号+命令+退出码；按行号定位，或用 `bash -x scripts/deploy_server.sh env 2>&1 \| tail -30` |
| 想复用已有环境却触发了安装 | `CONDA_ENV_NAME=<已有环境>` 且依赖探测不通过时，会调用 `create_conda_env.sh` 补齐。若不想动那个环境，去掉 `CONDA_ENV_NAME` 让它新建 |

---

## 14. 验收清单（部署完成的判定）

- [ ] `nvidia-smi` 显示只有 1 张卡被本服务占用（而非 4 张均沾）
- [ ] `scripts/verify.sh` 输出 `全部验证通过。`
- [ ] `/v1/documents` 中目标文档 `status=active` 且 `chunk_count>0`
- [ ] `/v1/query` 返回 `grounded=true` 且引用到「来源，第 N 页」
- [ ] 无 Key 访问 `/v1/query` 返回 `401`
- [ ] `/metrics` 返回 200 且含 `agentic_rag_http_requests_total`
- [ ] `systemctl is-enabled rag-agent` 为 `enabled`

未跑完本清单前，不要以「服务能启动」当作部署完成 —— 版本激活失败或空知识库同样能让进程健康起来。
