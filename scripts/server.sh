#!/usr/bin/env bash
# server.sh —— GPU 服务器端到端部署与运维（幂等，可重复执行）
#
# 目标环境：Linux x86_64 + NVIDIA 驱动 535.x（CUDA 12.2）+ 多卡（如 4×A40 46GB）+ Conda + curl
#
# 阶段（可单独执行，也可 all 一次跑完）：
#   env      1) 系统前置检查 + 创建/补齐 Conda 环境（torch 2.6.0 + cu124）
#   config   2) 生成 .env 并写入服务器推荐参数（单卡固定 / 7B 模型 / 并发 / MCP 白名单）
#   verify   3) 语法检查 + 全量测试 + 离线冒烟（不下载模型）
#   build    4) 下载模型并构建向量知识库
#   serve    5) 启动 FastAPI（0.0.0.0:8080），通过存活+就绪检查才返回
#   accept   6) 在线接口验收（健康 / 清单 / 问答 / 指标 / 鉴权边界）
#   systemd  7) 生成 systemd 常驻单元（开机自启，崩溃重拉）
#   docker   改用 Docker Compose 部署（DEPLOY_MODE=docker 的等价入口）
#   status | stop | logs | help
#
#   all = env → config → verify → build → serve → accept
#
# 可覆盖变量（前置赋值即可；显式传参优先于 .env）：
#   CONDA_ENV_NAME=rag-agent       TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
#   PYTHON_VERSION=3.10            LLM_MODEL=qwen/Qwen2.5-7B-Instruct
#   CUDA_VISIBLE_DEVICES=0         API_MAX_CONCURRENCY=2
#   APP_PORT=8080                  PUBLIC_HOST=（填域名/IP 会加入 MCP host 白名单）
#   DATA_DIR=data                  MODEL_CACHE_DIR=models
#   REQUIRE_CUDA=1                 DEVICE=auto
#   SKIP_ENV_CREATE=1              环境已就绪时跳过环境创建
#   SKIP_BUILD=1 / SKIP_EVAL=1 / SKIP_ACCEPT=1
#   INSTALL_SYSTEMD=1              systemd 阶段直接 sudo 安装并 enable
#
# 注意：本脚本不修改 requirements，不升级已有依赖；环境创建只发生在 env 阶段。
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"
cd "$PROJECT_DIR"
install_err_trap "部署中断"

# ---------------------------------------------------------------- 参数与默认值
# 显式传参（命令行前置赋值或环境变量）优先；.env 只提供默认值，不覆盖显式传参。
# 这里先把传入值原样记下来，避免被后面的 .env 读取覆盖。
CONDA_ENV_NAME_OVERRIDE="${CONDA_ENV_NAME-}"
DEPLOY_MODE_OVERRIDE="${DEPLOY_MODE-}"
REQUIRE_CUDA_OVERRIDE="${REQUIRE_CUDA-}"
PUBLIC_HOST_OVERRIDE="${PUBLIC_HOST-}"
LLM_MODEL_OVERRIDE="${LLM_MODEL-}"
CUDA_VISIBLE_DEVICES_OVERRIDE="${CUDA_VISIBLE_DEVICES-}"
API_MAX_CONCURRENCY_OVERRIDE="${API_MAX_CONCURRENCY-}"
APP_PORT_OVERRIDE="${APP_PORT-}"

# .env 中未显式传参时给出的默认值。
env_default() {
  local explicit="$1" key="$2" fallback="$3"
  if [[ -n "$explicit" ]]; then
    printf '%s' "$explicit"
    return 0
  fi
  local from_file
  from_file="$(env_get "$key")"
  printf '%s' "${from_file:-$fallback}"
}

PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
CONDA_CHANNEL="${CONDA_CHANNEL:-defaults}"
MIN_LINUX_DRIVER="450.80.02"
# 服务器侧默认装上 dev 依赖：评测脚本与测试都在同一环境里跑。
INSTALL_DEV="${INSTALL_DEV:-1}"

# 环境名优先级：显式传参 → .env → 当前已激活的非 base 环境 → rag-agent。
if [[ -n "$CONDA_ENV_NAME_OVERRIDE" ]]; then
  CONDA_ENV_NAME="$CONDA_ENV_NAME_OVERRIDE"
else
  CONDA_ENV_NAME="$(env_get CONDA_ENV_NAME)"
  if [[ -z "$CONDA_ENV_NAME" && -n "${CONDA_DEFAULT_ENV:-}" && "${CONDA_DEFAULT_ENV}" != "base" ]]; then
    CONDA_ENV_NAME="$CONDA_DEFAULT_ENV"
  fi
  CONDA_ENV_NAME="${CONDA_ENV_NAME:-rag-agent}"
fi

DEPLOY_MODE="$(env_default "$DEPLOY_MODE_OVERRIDE" DEPLOY_MODE conda)"
REQUIRE_CUDA="$(env_default "$REQUIRE_CUDA_OVERRIDE" REQUIRE_CUDA 1)"
PUBLIC_HOST="$(env_default "$PUBLIC_HOST_OVERRIDE" PUBLIC_HOST '')"
LLM_MODEL="$(env_default "$LLM_MODEL_OVERRIDE" LLM_MODEL qwen/Qwen2.5-7B-Instruct)"
CUDA_VISIBLE_DEVICES="$(env_default "$CUDA_VISIBLE_DEVICES_OVERRIDE" CUDA_VISIBLE_DEVICES 0)"
API_MAX_CONCURRENCY="$(env_default "$API_MAX_CONCURRENCY_OVERRIDE" API_MAX_CONCURRENCY 2)"
APP_PORT="$(env_default "$APP_PORT_OVERRIDE" APP_PORT 8080)"

DATA_DIR="${DATA_DIR:-data}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-models}"
APP_HOST="${APP_HOST:-0.0.0.0}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-300}"

SKIP_ENV_CREATE="${SKIP_ENV_CREATE:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_ACCEPT="${SKIP_ACCEPT:-0}"
INSTALL_SYSTEMD="${INSTALL_SYSTEMD:-0}"

UNIT_FILE="deploy/rag-agent.service"
export REQUIRE_CUDA

init_paths
mkdir -p "$DATA_DIR" "$MODEL_CACHE_DIR"

# ---------------------------------------------------------------- 环境创建
version_ge() { printf '%s\n%s\n' "$2" "$1" | sort -V -C; }

# 从零创建或补齐 Conda 环境。幂等：环境存在且依赖齐全时调用方不会走到这里。
# 刻意不使用 `conda run`：老版本 conda 不认 --no-capture-output，且 conda run 会缓冲
# 子进程输出，导致安装进度看不到；直接调用环境自带的解释器最稳。
create_or_repair_env() {
  require_cmd conda "请先安装 Miniconda/Anaconda"
  info "conda 版本：$(conda --version 2>&1)"

  if [[ "$REQUIRE_CUDA" == "1" ]]; then
    require_cmd nvidia-smi "REQUIRE_CUDA=1 时必须能读到 GPU"
    local driver
    driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | sed -n '1p' | tr -d '[:space:]' || true)"
    [[ -n "$driver" ]] || die "无法从 nvidia-smi 读取驱动版本，请手工确认 GPU 状态"
    if ! version_ge "$driver" "$MIN_LINUX_DRIVER"; then
      die "NVIDIA 驱动 $driver 低于 CUDA 11.x minor compatibility 要求 $MIN_LINUX_DRIVER"
    fi
    ok "检测到 NVIDIA 驱动：$driver"
  fi

  # 用 awk 读完整个输入再判定：`grep -q` 会提前退出让上游收到 SIGPIPE，pipefail 下整条管道被判失败。
  if conda env list 2>/dev/null | awk -v name="$CONDA_ENV_NAME" '$1 == name { found = 1 } END { exit(found ? 0 : 1) }'; then
    ok "环境已存在，复用：$CONDA_ENV_NAME"
  else
    info "创建 Conda 环境：$CONDA_ENV_NAME（Python $PYTHON_VERSION）"
    local create_args=(
      create -y -n "$CONDA_ENV_NAME" "python=$PYTHON_VERSION" pip
      --override-channels -c "$CONDA_CHANNEL"
    )
    if command -v mamba >/dev/null 2>&1; then
      info "使用 mamba 创建"
      mamba "${create_args[@]}"
    elif conda create --help 2>&1 | grep -q -- '--solver'; then
      info "优先使用 libmamba 求解器"
      if ! conda "${create_args[@]}" --solver=libmamba; then
        warn "libmamba 不可用，回退到 classic 求解器"
        conda "${create_args[@]}" --solver=classic
      fi
    else
      info "当前 Conda 不支持 --solver，使用 classic 求解器"
      conda "${create_args[@]}"
    fi
  fi

  local env_python
  env_python="$(resolve_env_python "$CONDA_ENV_NAME" || true)"
  if [[ -z "$env_python" || ! -x "$env_python" ]]; then
    die "无法定位 Conda 环境 $CONDA_ENV_NAME 的 Python 解释器，请检查：conda env list"
  fi
  info "环境解释器：$env_python"

  local env_version
  env_version="$("$env_python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [[ "$env_version" != "$PYTHON_VERSION" ]]; then
    die "环境 $CONDA_ENV_NAME 的 Python 为 $env_version，需要 $PYTHON_VERSION。请换 CONDA_ENV_NAME，或删除旧环境后重试。"
  fi

  if ! "$env_python" -m pip --version >/dev/null 2>&1; then
    info "检测到环境缺少 pip，正在自动修复"
    if ! "$env_python" -m ensurepip --upgrade; then
      warn "ensurepip 修复失败，改为强制重装 Conda pip 包"
      conda install -y -n "$CONDA_ENV_NAME" --force-reinstall \
        pip setuptools wheel --override-channels -c "$CONDA_CHANNEL"
    fi
  fi

  "$env_python" -m pip install --upgrade pip setuptools wheel

  # CUDA 11.1 对应的最后一个官方 PyTorch wheel 是 1.10.1，无法满足当前
  # LangGraph/MCP/Transformers 栈。这里安装 cu124 官方 wheel：自带 CUDA runtime，
  # 并在较老的 CUDA 11.x 驱动上通过 NVIDIA minor compatibility 运行。
  "$env_python" -m pip install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX_URL"
  "$env_python" -m pip install -r requirements.txt
  if [[ "$INSTALL_DEV" == "1" ]]; then
    "$env_python" -m pip install -r requirements-dev.txt
  fi
  "$env_python" -m pip check

  REQUIRE_CUDA="$REQUIRE_CUDA" "$env_python" - <<'PY'
import os

import fastapi  # noqa: F401
import langgraph  # noqa: F401
import mcp  # noqa: F401
import sentence_transformers  # noqa: F401
import torch
import transformers  # noqa: F401

print(f"Python/PyTorch 环境就绪: torch={torch.__version__}, runtime_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print("imports_ok=fastapi,langgraph,mcp,sentence_transformers,transformers")
if os.environ.get("REQUIRE_CUDA", "1") == "1":
    if not torch.cuda.is_available():
        raise SystemExit("ERROR: PyTorch 无法访问 CUDA；请检查驱动、GPU 权限和 CUDA_VISIBLE_DEVICES。")
    value = torch.ones(1, device="cuda") * 2
    torch.cuda.synchronize()
    print(f"gpu={torch.cuda.get_device_name(0)}, allocation_check={value.item()}")
PY
  ok "Conda 环境创建/校验完成：$CONDA_ENV_NAME"
}

# ---------------------------------------------------------------- 1) env
stage_env() {
  step "1/6 系统前置检查与环境准备"

  require_cmd curl "无法做健康检查与在线验收"
  require_cmd conda "请先安装 Miniconda/Anaconda"

  if command -v nvidia-smi >/dev/null 2>&1; then
    local driver gpu_count gpu0
    # 刻意不用 head -n 1：nvidia-smi 输出多行，head 提前退出会让 nvidia-smi 收到
    # SIGPIPE，pipefail 下整条管道被判失败；sed -n '1p' 读完再取首行。
    # 末尾 || true 兜底：探测失败不该中断整个部署。
    driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | sed -n '1p' | tr -d '[:space:]' || true)"
    gpu_count="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ' || true)"
    gpu0="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | sed -n '1p' | tr -d '\r' || true)"
    if [[ -n "$driver" ]]; then
      ok "NVIDIA 驱动 $driver ｜ GPU 数量 ${gpu_count:-?}"
      info "GPU0: ${gpu0:-未知}"
      info "本项目默认只用 CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES（单卡），其余卡留给别的任务"
    else
      warn "nvidia-smi 存在但查询不到驱动信息，请手工确认 GPU 状态后再继续"
    fi
  else
    warn "未找到 nvidia-smi：无 GPU 时请显式设置 REQUIRE_CUDA=0 DEVICE=cpu 后再启动"
  fi

  info "磁盘可用空间："
  df -h "$PROJECT_DIR" 2>/dev/null | tail -n 1 | sed 's/^/      /' || warn "无法读取磁盘信息（df 不可用），继续"
  info "模型缓存默认在 $MODEL_CACHE_DIR（BGE-M3 约 2.2GB + Reranker 约 1.1GB + 7B 生成模型约 15GB）"

  local create_env=0
  if [[ "$SKIP_ENV_CREATE" == "1" ]]; then
    warn "SKIP_ENV_CREATE=1，跳过环境创建"
  elif ! resolve_env_python "$CONDA_ENV_NAME" >/dev/null; then
    info "环境 $CONDA_ENV_NAME 不存在，将创建"
    create_env=1
  elif env_deps_ok "$(resolve_env_python "$CONDA_ENV_NAME" || true)"; then
    ok "复用 Conda 环境 $CONDA_ENV_NAME（依赖已齐，跳过创建与安装）"
  else
    info "环境 $CONDA_ENV_NAME 已存在但依赖不完整，将补齐（不会重建环境）"
    create_env=1
  fi

  if [[ "$create_env" == "1" ]]; then
    create_or_repair_env
  fi

  local py_bin
  py_bin="$(resolve_env_python "$CONDA_ENV_NAME" || true)"
  [[ -n "$py_bin" ]] || die "环境 $CONDA_ENV_NAME 不可用，请检查 conda env list"
  PYTHON_BIN="$py_bin"
  ok "解释器：$PYTHON_BIN"
  info "Python $("$PYTHON_BIN" -c 'import sys; print(sys.version.split()[0])' 2>/dev/null || echo '未知')"

  # torch / CUDA 自检。这是收尾探测，不能因为「读不到信息」把整个部署打断：
  # 用 || true 兜住（|| 列表里的前置命令不会触发 ERR trap），结果靠文本判断。
  local probe_out=""
  probe_out="$("$PYTHON_BIN" - <<'PY' 2>&1 || true
import torch

available = torch.cuda.is_available()
count = torch.cuda.device_count() if available else 0
name = torch.cuda.get_device_name(0) if available else "-"
print(f"torch={torch.__version__} | 编译 CUDA={torch.version.cuda} | cuda_available={available} | GPU 数={count} | 首卡={name}")
PY
)"
  if [[ "$probe_out" == *"cuda_available=True"* ]]; then
    ok "$probe_out"
  else
    warn "$probe_out"
    if [[ "$REQUIRE_CUDA" == "1" ]]; then
      die "REQUIRE_CUDA=1，但该环境无法使用 CUDA（详见上一行）。确认 torch 是 CUDA 轮子（--index-url .../cu124）；只想跑 CPU 就显式 REQUIRE_CUDA=0。"
    fi
    warn "该环境无法使用 CUDA（REQUIRE_CUDA=0，继续，将走 CPU）"
  fi
}

# ---------------------------------------------------------------- 2) config
stage_config() {
  step "2/6 生成 .env 并写入服务器参数"

  ensure_api_key

  env_set DEVICE "${DEVICE:-auto}"
  env_set CUDA_VISIBLE_DEVICES "$CUDA_VISIBLE_DEVICES"
  env_set LLM_MODEL "$LLM_MODEL"
  env_set LLM_BACKEND "${LLM_BACKEND:-local}"
  env_set API_MAX_CONCURRENCY "$API_MAX_CONCURRENCY"
  env_set MODEL_CACHE_DIR "$MODEL_CACHE_DIR"
  env_set CONDA_ENV_NAME "$CONDA_ENV_NAME"
  env_set REQUIRE_CUDA "$REQUIRE_CUDA"
  env_set APP_PORT "$APP_PORT"

  # PUBLIC_HOST 必须进 MCP host 白名单，否则经域名/IP 访问 /mcp/ 会被拒绝。
  local hosts origins
  hosts="$(env_get MCP_ALLOWED_HOSTS)"
  origins="$(env_get MCP_ALLOWED_ORIGINS)"
  [[ -n "$hosts" ]] || hosts="127.0.0.1:*,localhost:*,[::1]:*"
  if [[ -n "$PUBLIC_HOST" && ",${hosts}," != *",${PUBLIC_HOST},"* ]]; then
    hosts="${hosts},${PUBLIC_HOST},${PUBLIC_HOST}:*"
    env_set MCP_ALLOWED_HOSTS "$hosts"
    ok "MCP_ALLOWED_HOSTS 已加入 ${PUBLIC_HOST}"
  fi
  if [[ -n "$PUBLIC_HOST" && "$hosts" == *"$PUBLIC_HOST"* ]] && [[ "$origins" != *"$PUBLIC_HOST"* ]]; then
    info "若经 HTTPS 反向代理访问，浏览器 MCP 客户端还需把 https://${PUBLIC_HOST} 加进 MCP_ALLOWED_ORIGINS"
  fi

  chmod 600 .env
  ok ".env 已写入（权限 600）"
  echo
  grep -E '^(DEVICE|CUDA_VISIBLE_DEVICES|LLM_MODEL|LLM_BACKEND|API_MAX_CONCURRENCY|CONDA_ENV_NAME|REQUIRE_CUDA|APP_PORT|MCP_ALLOWED_HOSTS)=' .env | sed 's/^/      /'
}

# ---------------------------------------------------------------- 3) verify
stage_verify() {
  step "3/6 离线校验：语法 + 全量测试 + 端到端冒烟（不下载模型）"
  run_verify
  ok "离线校验全部通过"
}

# ---------------------------------------------------------------- 4) build
stage_build() {
  step "4/6 下载模型并构建向量知识库"
  if [[ "$SKIP_BUILD" == "1" ]]; then
    warn "SKIP_BUILD=1，跳过"
    return 0
  fi

  local pdf_count
  pdf_count="$( { find "$DATA_DIR" -maxdepth 1 -iname '*.pdf' 2>/dev/null || true; } | wc -l | tr -d ' ')"
  if [[ "$pdf_count" == "0" ]]; then
    warn "$DATA_DIR 下没有 PDF。请先上传，例如："
    echo "      scp ./你的文件.pdf <user>@<server>:$(pwd)/${DATA_DIR}/" >&2
    die "缺少知识库源文件"
  fi
  ok "$DATA_DIR 下共 $pdf_count 个 PDF"

  info "首次运行会从 ModelScope 下载缺失模型（BGE-M3 / BGE Reranker / $LLM_MODEL）"
  info "下载与向量化期间可另开终端观察：watch -n2 nvidia-smi"
  py create_db.py "$DATA_DIR" --replace
  print_document_records
  ok "向量库构建完成"
}

# ---------------------------------------------------------------- 5) serve
stage_serve() {
  step "5/6 启动 FastAPI 服务（${APP_HOST}:${APP_PORT}）"

  # serve 只复用、不创建环境（创建发生在 env 阶段）。这里显式失败，避免在环境不存在时
  # 悄悄回退到裸 python，最后以「服务进程已退出」这种看不懂的方式报错。
  local py_bin
  py_bin="$(resolve_env_python "$CONDA_ENV_NAME" || true)"
  if [[ -z "$py_bin" ]]; then
    die "未找到可复用的 Conda 环境：CONDA_ENV_NAME=${CONDA_ENV_NAME:-未设置}。先执行 bash scripts/server.sh env 创建，或用 CONDA_ENV_NAME=<已有环境> bash scripts/server.sh serve 指定。"
  fi
  PYTHON_BIN="$py_bin"

  if service_alive; then
    info "检测到 ${BASE_URL} 已有实例在响应，跳过启动"
    return 0
  fi

  local stale
  if stale="$(service_pid)"; then
    die "服务进程仍存在（PID=$stale），先执行 bash scripts/server.sh stop"
  fi
  rm -f "$PID_FILE"

  info "启动命令：$(locate_python) -m uvicorn api:app --host $APP_HOST --port $APP_PORT --workers 1"
  info "日志：$SERVICE_LOG"
  (
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
    export APP_PORT PUBLIC_HOST
    nohup "$(locate_python)" -m uvicorn api:app \
      --host "$APP_HOST" --port "$APP_PORT" --workers 1 \
      > "$SERVICE_LOG" 2>&1 &
    echo $! > "$PID_FILE"
  )

  local waited=0
  while (( waited < READY_TIMEOUT_SECONDS )); do
    if service_alive; then
      ok "存活检查通过（等待 ${waited}s）"
      break
    fi
    if ! service_pid >/dev/null; then
      tail -n 40 "$SERVICE_LOG" >&2 || true
      die "服务进程已退出，启动失败"
    fi
    sleep 2
    waited=$((waited + 2))
  done
  (( waited < READY_TIMEOUT_SECONDS )) || { tail -n 40 "$SERVICE_LOG" >&2 || true; die "等待服务就绪超时（${READY_TIMEOUT_SECONDS}s）"; }

  local code
  code="$(http_get /v1/health/ready)"
  [[ "$code" == "200" ]] || { cat "$BODY_FILE" 2>/dev/null; die "/v1/health/ready 返回 $code"; }
  ok "就绪检查通过（首次真实问答仍会懒加载 Embedding/Reranker/LLM，可能等待数十秒）"
  info "PID $(cat "$PID_FILE") ｜ 日志 tail -f $SERVICE_LOG"
}

# ---------------------------------------------------------------- 6) accept
stage_accept() {
  step "6/6 在线接口验收"
  local code
  [[ -n "$(api_key)" ]] || die ".env 中没有 API_KEY，先执行 bash scripts/server.sh config"

  info "1) 存活探针 /v1/health/live"
  code="$(probe_status "$BASE_URL/v1/health/live")"
  [[ "$code" == "200" ]] || die "返回 $code"
  ok "存活探针 200"

  info "2) 就绪探针 /v1/health/ready"
  code="$(http_get /v1/health/ready)"
  [[ "$code" == "200" ]] || { cat "$BODY_FILE"; die "返回 $code"; }
  ok "catalog.integrity=$(json_get catalog integrity) ｜ llm_backend=$(json_get llm_backend)"

  info "3) 文档清单 /v1/documents"
  code="$(http_get /v1/documents)"
  [[ "$code" == "200" ]] || die "返回 $code"
  ok "已索引文档 $(py -c "
import json
print('%d 个' % len(json.load(open('$BODY_FILE', encoding='utf-8'))['documents']))
")"

  info "4) 问答 /v1/query（真实模型推理，首次较慢）"
  local payload
  payload="$(py -c "
import json, sys
print(json.dumps({'question': sys.argv[1], 'history': []}, ensure_ascii=False))
" "$(probe_question)")"
  code="$(http_post /v1/query "$payload")"
  [[ "$code" == "200" ]] || { cat "$BODY_FILE"; die "/v1/query 返回 $code"; }
  ok "run_id=$(json_get run_id) ｜ grounded=$(json_get grounded) ｜ 时延=$(json_get latency_ms) ms"

  info "5) Prometheus 指标 /metrics"
  code="$(http_get /metrics)"
  [[ "$code" == "200" ]] || die "返回 $code"
  ok "指标端点 200"

  info "6) 鉴权边界：无 Key 访问 /v1/query 应被拒绝"
  code="$(probe_status -X POST -H 'Content-Type: application/json' -d "$payload" "$BASE_URL/v1/query")"
  if [[ "$code" == "401" ]]; then
    ok "无 Key 返回 401，API Key 门禁生效"
  else
    warn "无 Key 返回 $code（API_KEY 为空时门禁按设计关闭）"
  fi

  echo
  hr
  ok "验收通过。对外地址："
  info "OpenAPI: http://${PUBLIC_HOST:-<server>}:${APP_PORT}/docs"
  info "MCP:     http://${PUBLIC_HOST:-<server>}:${APP_PORT}/mcp/"
  info "Metrics: http://${PUBLIC_HOST:-<server>}:${APP_PORT}/metrics"
  info "API Key: $(api_key)"
}

# ---------------------------------------------------------------- 7) systemd
stage_systemd() {
  step "7/7 生成 systemd 常驻单元"
  local py_bin user
  py_bin="$(locate_python)"
  user="${SERVICE_USER:-$(id -un)}"
  mkdir -p deploy

  cat > "$UNIT_FILE" <<EOF
[Unit]
Description=Agentic RAG FastAPI service (uvicorn)
Documentation=file://${PROJECT_DIR}/docs/server-deploy-runbook.md
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${user}
WorkingDirectory=${PROJECT_DIR}
EnvironmentFile=-${PROJECT_DIR}/.env
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONDONTWRITEBYTECODE=1
ExecStart=${py_bin} -m uvicorn api:app --host ${APP_HOST} --port ${APP_PORT} --workers 1
Restart=on-failure
RestartSec=5
TimeoutStopSec=30
KillSignal=SIGINT
LimitNOFILE=65535
StandardOutput=append:${PROJECT_DIR}/${SERVICE_LOG}
StandardError=append:${PROJECT_DIR}/${SERVICE_LOG}

[Install]
WantedBy=multi-user.target
EOF

  ok "已生成 $UNIT_FILE（User=${user}，ExecStart 使用 ${py_bin}）"

  if [[ "$INSTALL_SYSTEMD" == "1" ]]; then
    info "安装到 /etc/systemd/system 并开机自启（需要 sudo）"
    sudo cp "$UNIT_FILE" /etc/systemd/system/rag-agent.service
    sudo systemctl daemon-reload
    sudo systemctl enable --now rag-agent
    sleep 3
    sudo systemctl --no-pager --lines=0 status rag-agent || true
    ok "已 enable --now rag-agent"
  else
    echo
    info "如需开机自启，手工执行："
    echo "      sudo cp $UNIT_FILE /etc/systemd/system/rag-agent.service"
    echo "      sudo systemctl daemon-reload"
    echo "      sudo systemctl enable --now rag-agent"
    echo "      systemctl status rag-agent"
    info "使用 systemd 后请先停止手工启动的进程：bash scripts/server.sh stop"
  fi
}

# ---------------------------------------------------------------- docker
stage_docker() {
  step "Docker Compose 部署"
  require_cmd docker "docker 未安装或不在 PATH 中"
  docker compose version >/dev/null 2>&1 || die "docker compose 插件不可用"
  ensure_api_key

  docker compose config --quiet
  info "构建并启动：docker compose up --build -d --remove-orphans"
  docker compose up --build -d --remove-orphans

  # 两阶段等待：先存活（不鉴权），再就绪（带 Key）。
  local deadline
  deadline=$((SECONDS + READY_TIMEOUT_SECONDS))
  info "等待存活检查：$BASE_URL/v1/health/live"
  until service_alive; do
    if (( SECONDS >= deadline )); then
      docker compose logs --tail=80 agentic-rag >&2 || true
      die "服务未在 ${READY_TIMEOUT_SECONDS}s 内通过存活检查"
    fi
    sleep 2
  done
  ok "存活检查通过"

  info "等待就绪检查：$BASE_URL/v1/health/ready"
  until [[ "$(http_get /v1/health/ready)" == "200" ]]; do
    if (( SECONDS >= deadline )); then
      docker compose logs --tail=80 agentic-rag >&2 || true
      die "服务未在 ${READY_TIMEOUT_SECONDS}s 内通过就绪检查"
    fi
    sleep 2
  done
  ok "就绪检查通过"

  echo
  info "REST API: http://<server>:${APP_PORT}/docs"
  info "MCP:      http://<server>:${APP_PORT}/mcp/"
  info "Metrics:  http://<server>:${APP_PORT}/metrics"
  info "查看日志：docker compose logs -f agentic-rag"
  info "停止服务：docker compose down"
}

# ---------------------------------------------------------------- 运维
cmd_status() {
  hr
  echo "项目：  $PROJECT_DIR"
  echo "环境：  $CONDA_ENV_NAME"
  echo "解释器：$(locate_python 2>/dev/null || echo '未就绪')"
  echo "地址：  $BASE_URL"
  local pid
  if service_alive; then
    pid="$(service_pid || legacy_service_pid || echo '-')"
    echo "服务：  运行中（HTTP 200，PID $pid）"
  else
    echo "服务：  未运行"
  fi
  if command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files 2>/dev/null | grep -q '^rag-agent.service'; then
    echo "systemd：$(systemctl is-active rag-agent 2>/dev/null || echo unknown) / $(systemctl is-enabled rag-agent 2>/dev/null || echo unknown)"
  fi
  if [[ -x "$(locate_python 2>/dev/null || echo /nonexistent)" ]]; then
    print_document_records
  fi
  hr
}

cmd_stop() {
  stop_service
  if command -v systemctl >/dev/null 2>&1 && systemctl is-active rag-agent >/dev/null 2>&1; then
    info "停止 systemd 单元 rag-agent"
    sudo systemctl stop rag-agent || true
  fi
}

cmd_logs() { tail -n "${LINES:-120}" -f "$SERVICE_LOG"; }

cmd_help() { sed -n '2,/^set -/p' "${BASH_SOURCE[0]}" | sed '$d'; }

# ---------------------------------------------------------------- 主流程
case "${1:-all}" in
  env)     stage_env ;;
  config)  stage_config ;;
  verify)  stage_verify ;;
  build)   stage_build ;;
  serve)   stage_serve ;;
  accept)  stage_accept ;;
  systemd) stage_systemd ;;
  docker)  stage_docker ;;
  all)
    if [[ "${DEPLOY_MODE}" == "docker" ]]; then
      stage_docker
      exit 0
    fi
    echo "############################################################"
    echo "# RAG 知识库问答 Agent —— 服务器部署"
    echo "# 项目：  $PROJECT_DIR"
    echo "# 环境：  $CONDA_ENV_NAME（torch $TORCH_VERSION / $TORCH_INDEX_URL）"
    echo "# 模型：  $LLM_MODEL ｜ 可见 GPU：$CUDA_VISIBLE_DEVICES"
    echo "# 地址：  ${APP_HOST}:${APP_PORT}"
    echo "############################################################"
    stage_env
    stage_config
    if [[ "$SKIP_EVAL" == "1" ]]; then warn "SKIP_EVAL=1，跳过离线校验"; else stage_verify; fi
    stage_build
    stage_serve
    if [[ "$SKIP_ACCEPT" == "1" ]]; then warn "SKIP_ACCEPT=1，跳过在线验收"; else stage_accept; fi
    echo
    echo "部署完成。建议继续："
    echo "  bash scripts/server.sh systemd     # 开机自启"
    echo "  bash scripts/server.sh status      # 状态总览"
    echo "  bash scripts/local.sh eval         # 固定问题集评测"
    ;;
  status) cmd_status ;;
  stop)   cmd_stop ;;
  logs)   cmd_logs ;;
  -h|--help|help) cmd_help ;;
  *) echo "未知参数：$1" >&2; cmd_help >&2; exit 1 ;;
esac
