# A40 服务器 Conda 环境配置

## 一、目标环境与选型依据

| 项目 | 值 |
|---|---|
| GPU | 4× NVIDIA A40（Ampere，sm_86，46GB/卡，支持 bf16） |
| 驱动 | 535.183.01（CUDA 12.2） |
| Python | 3.10 |
| PyTorch | 2.6.0 |
| CUDA 轮子 | **cu124**（备选 cu126 / 兜底 cu118） |

选型说明：

- 已确认 `download.pytorch.org` 上 `torch 2.6.0` 只提供 **cu118、cu124、cu126** 三种 Linux x86_64 轮子，没有 cu121。
- 驱动 535 对应 CUDA 12.2。按 CUDA 12.x 的 minor version compatibility 规则（驱动 ≥ 525.60.13 即可运行任意 12.x 运行时），**cu124 可直接使用**，无需升级驱动。
- A40 支持 bf16，`ModelRuntime` 会自动走 `torch.bfloat16`（`torch.cuda.is_bf16_supported()` 为真），比 fp16 更稳，无需手工改代码。
- 项目 `requirements.txt` 约束为 `torch>=2.6,<3`，与 2.6.0 一致。

---

## 二、方式 A（推荐）：复用项目自带脚本，只覆盖索引地址

`create_conda_env.sh` 已包含建环境 → 装 PyTorch → 装 `requirements.txt` → `pip check` → 导入与 CUDA 实算自检，并会校验驱动版本（要求 ≥ 450.80.02，535 满足）。

```bash
# 1) 把项目放到服务器（二选一）
#    rsync -avz --exclude .backup --exclude runs --exclude reports \
#      ./rag-agent-upgrade/ user@your-server:/data/apps/rag-agent-upgrade/
#    或直接在服务器上用 git clone / 解压压缩包

cd /data/apps/rag-agent-upgrade
chmod +x create_conda_env.sh run_server.sh

# 2) 建环境（Python 3.10 + torch 2.6.0 cu124 + 全部依赖，含测试依赖）
CONDA_ENV_NAME=rag-agent \
TORCH_VERSION=2.6.0 \
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
PYTHON_VERSION=3.10 \
REQUIRE_CUDA=1 \
INSTALL_DEV=1 \
./create_conda_env.sh
```

脚本最后会打印 `torch=2.6.0+cu124`、`cuda_available=True`、显卡名与一次 GPU 实算结果，作为环境可用证据。

---

## 三、方式 B：手工逐条执行（需要自己控制每一步时）

```bash
# 1) 建环境
conda create -y -n rag-agent python=3.10 pip
conda activate rag-agent

# 2) 基础工具链
python -m pip install --upgrade pip setuptools wheel

# 3) PyTorch（务必带 --index-url，否则会装到 CPU 轮子）
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 4) 项目依赖
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt    # 仅测试/开发需要

# 5) 一致性检查
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
```

---

## 四、环境校验（装完必跑）

```bash
conda activate rag-agent
cd /data/apps/rag-agent-upgrade

# 依赖 + GPU 自检
bash scripts/run_local.sh check

# 离线端到端验证：语法检查 + 全量测试 + 不下载模型的冒烟
bash scripts/verify.sh
```

期望结果：`compileall 通过`、`pytest 44 passed`、`离线冒烟全部通过`，且 `check` 打印 `GPU: NVIDIA A40 / 46.0 GB`。

---

## 五、A40 上的 `.env` 调参建议

```bash
cp .env.example .env
```

| 变量 | 建议值 | 原因 |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | `0` | 不设置时 `device_map="auto"` 会把模型切到 4 张卡上；固定单卡便于隔离与排障。要用多卡再显式放开 |
| `DEVICE` | `auto` | 自动选 cuda；bf16 由代码自动判定，无需手工指定 |
| `LLM_MODEL` | `qwen/Qwen2.5-7B-Instruct` | A40 46GB 单卡可轻松容纳（bf16 约 15GB）。**基线与 Agent 评测必须用同一模型** |
| `LLM_BACKEND` | `local` | 若要拆成独立推理服务，改成 `openai_compatible` 并配 `LLM_API_BASE`（vLLM 等） |
| `API_MAX_CONCURRENCY` | `2`~`4` | 显存充裕，可放开并发；评测时建议回到 `1` 保持可比 |
| `MAX_UPLOAD_BYTES` | 保持默认 | 服务端可放宽，但不要超过反向代理的上传限制 |
| `REQUIRE_CUDA` | `1` | `run_server.sh` 启动前会做 CUDA 检查 |

启动服务：

```bash
# 只复用已有环境，不创建环境、不安装依赖
CONDA_ENV_NAME=rag-agent REQUIRE_CUDA=1 bash run_server.sh
```

模型规模与显存对照（bf16，单卡 A40 46GB）：

| 模型 | 显存占用 | 结论 |
|---|---:|---|
| Qwen2.5-1.5B | ~3 GB | 本机复现用 |
| Qwen2.5-7B | ~15 GB | **服务器推荐**，余量充足 |
| Qwen2.5-14B | ~28 GB | 可用，并发要收敛 |
| Qwen2.5-32B | ~65 GB | 单卡放不下，需多卡或量化 |

Embedding（BGE-M3，fp32）约 2.3GB、Reranker（bge-reranker-base）约 0.6GB，与生成模型共享同一张卡，上面已计入余量。

---

## 六、常见问题

| 现象 | 原因与处理 |
|---|---|
| `torch.cuda.is_available()` 为 `False` | 装到了 CPU 轮子。确认安装时带了 `--index-url https://download.pytorch.org/whl/cu124`，然后 `pip uninstall -y torch && pip install torch==2.6.0 --index-url ...` |
| `CUDA error: no kernel image is available` | 轮子与架构不匹配。本项目走的 cu124 官方轮子已包含 sm_86，如出现请确认没有装 `+cpu` 版本 |
| 想彻底回避 12.x 兼容性讨论 | 改用 cu118：把上面所有 `cu124` 换成 `cu118`。功能一致，只是少了 12.x 的新特性 |
| 下载模型慢 | `MODEL_CACHE_DIR` 指向大盘目录（默认 `models/`）。首次建库需从 ModelScope 拉取 BGE-M3 + Reranker + 生成模型 |
| 多卡机器上显存被摊开 | 必须设 `CUDA_VISIBLE_DEVICES=0`；不设时 `device_map="auto"` 会跨卡切分 |
| `conda run` 报 `unrecognized arguments: --no-capture-output` | 旧版 conda（< 4.9）不支持该参数。`create_conda_env.sh` 与 `run_server.sh` 已改为**直接从 `conda env list` 解析环境前缀、调用环境自带解释器**，完全不使用 `conda run`；升级到最新脚本即可，无需升级 conda |

---

## 七、建库、评测与服务（环境就绪后）

> 从零到服务可用的**分阶段可复制命令**，见 [服务器部署流程命令](server-deploy-runbook.md)；也可直接执行幂等脚本 `bash scripts/deploy_server.sh all`。

```bash
conda activate rag-agent
cd /data/apps/rag-agent-upgrade

# 一键全流程（自检→配置→数据→离线自检→下载模型建库→评测→起服务→接口验收）
bash scripts/run_all.sh

# 或分步
bash scripts/run_local.sh build     # 建/重建向量知识库
bash scripts/run_local.sh eval      # 固定问题集评测，报告见 reports/
bash scripts/run_server.sh          # 生产式启动（CUDA 检查 + 健康检查）
bash scripts/run_all.sh status      # 查看服务与知识库状态
bash scripts/run_all.sh stop        # 停止服务
```
