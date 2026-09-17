#!/usr/bin/env bash
# deploy_server.sh —— GPU 服务器端到端部署（幂等，可重复执行）
#
# 目标环境：Linux x86_64 + NVIDIA 驱动 535.x（CUDA 12.2）+ 4×A40 46GB + Conda + curl
#
# 阶段（可单独执行，也可 all 一次跑完）：
#   env      1) 系统前置检查 + 创建/复用 Conda 环境（torch 2.6.0 + cu124）
#   config   2) 生成 .env 并写入 A40 推荐参数（单卡固定 / 7B 模型 / 并发 / MCP 白名单）
#   verify   3) 语法检查 + 全量测试 + 离线冒烟（不下载模型）
#   build    4) 下载模型并构建向量知识库
#   serve    5) 启动 FastAPI（0.0.0.0:8080），通过存活+就绪检查才返回
#   accept   6) 在线接口验收（健康 / 清单 / 问答 / 指标 / 鉴权边界）
#   systemd  7) 生成 systemd 常驻单元（开机自启，崩溃重拉）
#   status | stop | logs | help
#
#   all = env → config → verify → build → serve → accept
#
# 可覆盖变量（前置赋值即可）：
#   CONDA_ENV_NAME=rag-agent       TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
#   PYTHON_VERSION=3.10            LLM_MODEL=qwen/Qwen2.5-7B-Instruct
#   CUDA_VISIBLE_DEVICES=0         API_MAX_CONCURRENCY=2
#   APP_PORT=8080                  PUBLIC_HOST=（填域名/IP 会加入 MCP host 白名单）
#   DATA_DIR=data                  MODEL_CACHE_DIR=models
#   SKIP_ENV_CREATE=1              环境已就绪时跳过环境创建
#   SKIP_BUILD=1 / SKIP_EVAL=1 / SKIP_ACCEPT=1
#   INSTALL_SYSTEMD=1              systemd 阶段直接 sudo 安装并 enable
#
# 注意：本脚本不修改 requirements，不升级已有依赖；环境创建只发生在 env 阶段。
set -Eeuo pipefail

# ---------------------------------------------------------------- 失败可见性
# set -e 会让失败的命令直接退出，但「命令替换里的管道」「df 这类探测」失败时自身
# 不打印任何可读信息，表现就是「跑到一半静默回到提示符」，非常难定位。
# 这里装上 ERR trap：只在真正失败时触发（|| / if 条件里的预期失败不会触发），
# 把文件、行号、失败命令、退出码一起打出来，杜绝静默中断。
trap 'err_status=$?; err_line=$LINENO; err_cmd=$BASH_COMMAND; if [[ "$err_line" != "${ERR_SEEN_LINE:-}" ]]; then ERR_SEEN_LINE="$err_line"; printf "\n   ❌ 部署中断：第 %s 行执行失败（退出码 %s）\n" "$err_line" "$err_status" >&2; printf "      失败命令：%s\n" "$err_cmd" >&2; printf "      文件位置：%s\n" "${BASH_SOURCE[0]}" >&2; printf "      查看该行：sed -n %sp %s\n" "$err_line" "${BASH_SOURCE[0]}" >&2; fi; true' ERR

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-rag-agent}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
LLM_MODEL="${LLM_MODEL:-qwen/Qwen2.5-7B-Instruct}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
API_MAX_CONCURRENCY="${API_MAX_CONCURRENCY:-2}"
APP_PORT="${APP_PORT:-8080}"
PUBLIC_HOST="${PUBLIC_HOST:-}"
DATA_DIR="${DATA_DIR:-data}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-models}"
APP_HOST="${APP_HOST:-0.0.0.0}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-300}"

SKIP_ENV_CREATE="${SKIP_ENV_CREATE:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_ACCEPT="${SKIP_ACCEPT:-0}"
INSTALL_SYSTEMD="${INSTALL_SYSTEMD:-0}"

BASE_URL="http://127.0.0.1:${APP_PORT}"
RUN_DIR="runs"
PID_FILE="$RUN_DIR/uvicorn.pid"
SERVICE_LOG="$RUN_DIR/uvicorn.log"
UNIT_FILE="deploy/rag-agent.service"
BODY_FILE="$RUN_DIR/.deploy_body.json"
mkdir -p "$RUN_DIR" "$DATA_DIR"

# ---------------------------------------------------------------- 输出
hr()   { printf '%s\n' "------------------------------------------------------------"; }
step() { echo; hr; echo "▶ $*"; hr; }
ok()   { echo "   ✅ $*"; }
info() { echo "   ·  $*"; }
warn() { echo "   ⚠️  $*"; }
die()  { echo "   ❌ $*" >&2; exit 1; }

# ---------------------------------------------------------------- 解释器定位
# 用 conda env list 解析环境前缀，比 conda run 更稳，也不要求环境已激活。
# 取每行最后一个字段作路径，可兼容「已激活」行结尾的 * 标记。
resolve_env_python() {
  local env_name="$1" prefix candidate
  [[ -n "$env_name" ]] || return 1
  command -v conda >/dev/null 2>&1 || return 1
  # sed -n '1p' 而不是 head -n 1：head 提前退出会让上游命令收到 SIGPIPE，
  # 在 set -o pipefail 下整条管道被判失败；末尾 || true 再兜一层。
  prefix="$(conda env list 2>/dev/null \
    | awk -v name="$env_name" '$1 == name { print $NF }' \
    | sed -n '1p' | tr -d '\r' || true)"
  prefix="${prefix//\\//}"
  for candidate in "$prefix/bin/python" "$prefix/python.exe"; do
    if [[ -n "$prefix" && -x "$candidate" ]]; then
      printf '%s' "$candidate"
      return 0
    fi
  done
  return 1
}

# 环境已存在时，快速判断依赖是否装齐（只读检查，不安装任何东西）。
# 不齐时交给 create_conda_env.sh 补齐 —— 该脚本复用已存在环境，不会重建。
env_deps_ok() {
  local py
  py="$(resolve_env_python "$1" || true)"
  [[ -n "$py" && -x "$py" ]] || return 1
  "$py" - <<'PY' >/dev/null 2>&1
import importlib.util

required = (
    "fastapi", "uvicorn", "mcp", "langgraph", "langchain_chroma",
    "langchain_community", "langchain_text_splitters", "httpx",
    "torch", "modelscope", "sentence_transformers", "transformers",
    "dotenv", "pypdf", "multipart", "chromadb", "streamlit", "pytest",
)
missing = [name for name in required if importlib.util.find_spec(name) is None]
raise SystemExit(1 if missing else 0)
PY
}

ENV_PYTHON=""
locate_python() {
  if [[ -n "$ENV_PYTHON" ]]; then
    printf '%s' "$ENV_PYTHON"
    return 0
  fi
  if ENV_PYTHON="$(resolve_env_python "$CONDA_ENV_NAME")"; then
    printf '%s' "$ENV_PYTHON"
    return 0
  fi
  if [[ -x "${PYTHON_BIN:-}" ]]; then
    ENV_PYTHON="$PYTHON_BIN"
    printf '%s' "$ENV_PYTHON"
    return 0
  fi
  die "未找到 Conda 环境 $CONDA_ENV_NAME 的解释器。先执行：bash scripts/deploy_server.sh env"
}

# ---------------------------------------------------------------- .env 读写
env_get() {
  [[ -f .env ]] || { printf ''; return 0; }
  sed -n "s/^$1=//p" .env | tail -n 1 | tr -d '\r'
}

# 幂等写入：存在则替换，不存在则追加。转义 sed 的 & 与 | ，避免值里的特殊字符被吞。
env_set() {
  local key="$1" value="$2" escaped
  [[ -f .env ]] || cp .env.example .env
  escaped="${value//\\/\\\\}"
  escaped="${escaped//&/\\&}"
  escaped="${escaped//|/\\|}"
  if grep -qE "^[[:space:]]*${key}=" .env; then
    sed -i "s|^[[:space:]]*${key}=.*|${key}=${escaped}|" .env
  else
    printf '\n%s=%s\n' "$key" "$value" >> .env
  fi
}

api_key() { env_get API_KEY; }

http_code() { # http_code <method> <path> [json-body]
  local method="$1" path="$2" body="${3:-}"
  if [[ "$method" == "GET" ]]; then
    curl -sS -o "$BODY_FILE" -w '%{http_code}' \
      -H "X-API-Key: $(api_key)" "$BASE_URL$path" 2>/dev/null || echo "000"
  else
    curl -sS -o "$BODY_FILE" -w '%{http_code}' -X POST \
      -H "X-API-Key: $(api_key)" -H 'Content-Type: application/json' \
      -d "$body" "$BASE_URL$path" 2>/dev/null || echo "000"
  fi
}

py() { "$(locate_python)" "$@"; }

# ---------------------------------------------------------------- 1) env
stage_env() {
  step "1/6 系统前置检查与环境准备"

  command -v curl >/dev/null 2>&1 || die "缺少 curl，无法做健康检查与在线验收"
  command -v conda >/dev/null 2>&1 || die "缺少 conda，请先安装 Miniconda/Anaconda"

  if command -v nvidia-smi >/dev/null 2>&1; then
    local driver gpu_count gpu0
    # 刻意不用 head -n 1：nvidia-smi 输出多行，head 提前退出会让 nvidia-smi 收到
    # SIGPIPE，在 set -o pipefail 下整条管道被判失败；sed -n '1p' 读完再取首行。
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
    info "创建 Conda 环境：$CONDA_ENV_NAME（Python $PYTHON_VERSION, torch $TORCH_VERSION, $TORCH_INDEX_URL）"
    create_env=1
  elif env_deps_ok "$CONDA_ENV_NAME"; then
    ok "复用 Conda 环境 $CONDA_ENV_NAME（依赖已齐，跳过创建与安装）"
  else
    info "Conda 环境 $CONDA_ENV_NAME 已存在但依赖不完整，调用 create_conda_env.sh 补齐"
    info "（create_conda_env.sh 复用已存在环境，不会重建环境）"
    create_env=1
  fi

  if [[ "$create_env" == "1" ]]; then
    CONDA_ENV_NAME="$CONDA_ENV_NAME" PYTHON_VERSION="$PYTHON_VERSION" \
    TORCH_VERSION="$TORCH_VERSION" TORCH_INDEX_URL="$TORCH_INDEX_URL" \
    REQUIRE_CUDA="${REQUIRE_CUDA:-1}" INSTALL_DEV=1 \
      bash create_conda_env.sh
  fi

  ENV_PYTHON="$(resolve_env_python "$CONDA_ENV_NAME" || true)"
  [[ -n "$ENV_PYTHON" ]] || die "环境 $CONDA_ENV_NAME 不可用，请检查 conda env list"
  ok "解释器：$ENV_PYTHON"
  info "Python $("$(locate_python)" -c 'import sys; print(sys.version.split()[0])' 2>/dev/null || echo '未知')"

  # torch / CUDA 自检。这是收尾探测，不能因为「读不到信息」把整个部署打断：
  # 用 || true 兜住（|| 列表里的前置命令不会触发 ERR trap），结果靠文本判断。
  local probe_out=""
  probe_out="$("$(locate_python)" - <<'PY' 2>&1 || true
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
    if [[ "${REQUIRE_CUDA:-1}" == "1" ]]; then
      die "REQUIRE_CUDA=1，但该环境无法使用 CUDA（详见上一行）。确认 torch 是 CUDA 轮子（--index-url .../cu124）；只想跑 CPU 就显式 REQUIRE_CUDA=0。"
    fi
    warn "该环境无法使用 CUDA（REQUIRE_CUDA=0，继续，将走 CPU）"
  fi
}

# ---------------------------------------------------------------- 2) config
stage_config() {
  step "2/6 生成 .env 并写入服务器参数"

  [[ -f .env ]] || { cp .env.example .env; ok "已从 .env.example 生成 .env"; }

  local key
  key="$(api_key)"
  if [[ -z "$key" ]]; then
    key="$(openssl rand -hex 32 2>/dev/null || "$(locate_python)" -c 'import secrets; print(secrets.token_hex(32))')"
    env_set API_KEY "$key"
    ok "已生成强随机 API_KEY"
  else
    info "沿用 .env 中已有 API_KEY（前 8 位 ${key:0:8}…）"
  fi

  env_set DEVICE auto
  env_set CUDA_VISIBLE_DEVICES "$CUDA_VISIBLE_DEVICES"
  env_set LLM_MODEL "$LLM_MODEL"
  env_set LLM_BACKEND local
  env_set API_MAX_CONCURRENCY "$API_MAX_CONCURRENCY"
  env_set MODEL_CACHE_DIR "$MODEL_CACHE_DIR"
  env_set CONDA_ENV_NAME "$CONDA_ENV_NAME"
  env_set REQUIRE_CUDA "${REQUIRE_CUDA:-1}"

  # PUBLIC_HOST 必须进 MCP host 白名单，否则经域名访问 /mcp/ 会被拒绝。
  local hosts origins
  hosts="$(env_get MCP_ALLOWED_HOSTS)"
  origins="$(env_get MCP_ALLOWED_ORIGINS)"
  if [[ -z "$hosts" ]]; then
    hosts="127.0.0.1:*,localhost:*,[::1]:*"
  fi
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
  CONDA_ENV_NAME="$CONDA_ENV_NAME" PYTHON_BIN="$(locate_python)" bash scripts/verify.sh
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
  "$(locate_python)" create_db.py "$DATA_DIR" --replace

  "$(locate_python)" - <<'PY'
from agentic_rag import Settings
from agentic_rag.knowledge_base import KnowledgeBase
from agentic_rag.models import ModelRuntime

settings = Settings.from_env()
for record in KnowledgeBase(settings, ModelRuntime(settings)).document_records():
    print(f"   ·  {record.source}: {record.status} / {record.chunk_count} 块 / version={(record.active_version_id or '-')[:12]}")
PY
  ok "向量库构建完成"
}

# ---------------------------------------------------------------- 5) serve
stage_serve() {
  step "5/6 启动 FastAPI 服务（${APP_HOST}:${APP_PORT}）"

  if [[ "$(curl -sS -o /dev/null -w '%{http_code}' "$BASE_URL/v1/health/live" 2>/dev/null || echo 000)" == "200" ]]; then
    info "检测到 ${BASE_URL} 已有实例在响应，跳过启动"
    return 0
  fi

  if [[ -f "$PID_FILE" ]]; then
    local stale
    stale="$(tr -d ' \r\n' < "$PID_FILE")"
    if kill -0 "$stale" >/dev/null 2>&1; then
      die "服务进程仍存在（PID=$stale），先执行 bash scripts/deploy_server.sh stop"
    fi
    rm -f "$PID_FILE"
  fi

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
    if [[ "$(curl -sS -o /dev/null -w '%{http_code}' "$BASE_URL/v1/health/live" 2>/dev/null || echo 000)" == "200" ]]; then
      ok "存活检查通过（等待 ${waited}s）"
      break
    fi
    if ! kill -0 "$(tr -d ' \r\n' < "$PID_FILE" 2>/dev/null)" >/dev/null 2>&1; then
      tail -n 40 "$SERVICE_LOG" >&2 || true
      die "服务进程已退出，启动失败"
    fi
    sleep 2
    waited=$((waited + 2))
  done
  (( waited < READY_TIMEOUT_SECONDS )) || { tail -n 40 "$SERVICE_LOG" >&2 || true; die "等待服务就绪超时（${READY_TIMEOUT_SECONDS}s）"; }

  local code
  code="$(http_code GET /v1/health/ready)"
  [[ "$code" == "200" ]] || { cat "$BODY_FILE" 2>/dev/null; die "/v1/health/ready 返回 $code"; }
  ok "就绪检查通过（首次真实问答仍会懒加载 Embedding/Reranker/LLM，可能等待数十秒）"
  info "PID $(cat "$PID_FILE") ｜ 日志 tail -f $SERVICE_LOG"
}

# ---------------------------------------------------------------- 6) accept
stage_accept() {
  step "6/6 在线接口验收"
  local key
  key="$(api_key)"
  [[ -n "$key" ]] || die ".env 中没有 API_KEY，先执行 bash scripts/deploy_server.sh config"

  local code
  info "1) 存活探针 /v1/health/live"
  code="$(curl -sS -o /dev/null -w '%{http_code}' "$BASE_URL/v1/health/live" 2>/dev/null || echo 000)"
  [[ "$code" == "200" ]] || die "返回 $code"
  ok "存活探针 200"

  info "2) 就绪探针 /v1/health/ready"
  code="$(http_code GET /v1/health/ready)"
  [[ "$code" == "200" ]] || { cat "$BODY_FILE"; die "返回 $code"; }
  ok "catalog.integrity=$(py -c "import json;print(json.load(open('$BODY_FILE',encoding='utf-8'))['catalog']['integrity'])") ｜ llm_backend=$(py -c "import json;print(json.load(open('$BODY_FILE',encoding='utf-8'))['llm_backend'])")"

  info "3) 文档清单 /v1/documents"
  code="$(http_code GET /v1/documents)"
  [[ "$code" == "200" ]] || die "返回 $code"
  ok "$(py -c "import json;d=json.load(open('$BODY_FILE',encoding='utf-8'))['documents'];print('已索引文档 %d 个' % len(d))")"

  info "4) 问答 /v1/query（真实模型推理，首次较慢）"
  local payload
  payload="$(py - <<'PY'
import json
from pathlib import Path

question = "知识库里有哪些文档？"
path = Path("eval/dataset.example.jsonl")
if path.exists():
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if item.get("expected_sources"):
                question = item["question"]
                break
print(json.dumps({"question": question, "history": []}, ensure_ascii=False))
PY
)"
  code="$(http_code POST /v1/query "$payload")"
  [[ "$code" == "200" ]] || { cat "$BODY_FILE"; die "/v1/query 返回 $code"; }
  ok "run_id=$(py -c "import json;print(json.load(open('$BODY_FILE',encoding='utf-8'))['run_id'])") ｜ grounded=$(py -c "import json;print(json.load(open('$BODY_FILE',encoding='utf-8'))['grounded'])") ｜ 时延=$(py -c "import json;print(json.load(open('$BODY_FILE',encoding='utf-8'))['latency_ms'])") ms"

  info "5) Prometheus 指标 /metrics"
  code="$(http_code GET /metrics)"
  [[ "$code" == "200" ]] || die "返回 $code"
  ok "指标端点 200"

  info "6) 鉴权边界：无 Key 访问 /v1/query 应被拒绝"
  code="$(curl -sS -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' \
    -d "$payload" "$BASE_URL/v1/query" 2>/dev/null || echo 000)"
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
    info "使用 systemd 后请先停止手工启动的进程：bash scripts/deploy_server.sh stop"
  fi
}

# ---------------------------------------------------------------- 运维
cmd_status() {
  hr
  echo "项目：  $PROJECT_DIR"
  echo "环境：  $CONDA_ENV_NAME"
  echo "解释器：$(locate_python 2>/dev/null || echo '未就绪')"
  echo "地址：  $BASE_URL"
  if [[ "$(curl -sS -o /dev/null -w '%{http_code}' "$BASE_URL/v1/health/live" 2>/dev/null || echo 000)" == "200" ]]; then
    echo "服务：  运行中（HTTP 200，PID $(tr -d ' \r\n' < "$PID_FILE" 2>/dev/null || echo '-')）"
  else
    echo "服务：  未运行"
  fi
  if command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files 2>/dev/null | grep -q '^rag-agent.service'; then
    echo "systemd：$(systemctl is-active rag-agent 2>/dev/null || echo unknown) / $(systemctl is-enabled rag-agent 2>/dev/null || echo unknown)"
  fi
  if [[ -x "$(locate_python 2>/dev/null)" ]]; then
    "$(locate_python)" - <<'PY'
from agentic_rag import Settings
from agentic_rag.knowledge_base import KnowledgeBase
from agentic_rag.models import ModelRuntime

settings = Settings.from_env()
records = KnowledgeBase(settings, ModelRuntime(settings)).document_records()
if not records:
    print("知识库：空（先执行 bash scripts/deploy_server.sh build）")
for record in records:
    print(f"知识库：{record.source} | {record.status} | {record.chunk_count} 块 | version={(record.active_version_id or '-')[:12]}")
PY
  fi
  hr
}

cmd_stop() {
  local stopped=0
  for f in "$PID_FILE" "$RUN_DIR/service.pid"; do
    [[ -f "$f" ]] || continue
    local pid
    pid="$(tr -d ' \r\n' < "$f")"
    # PID 会被系统回收复用：只凭 kill -0 通过就下手，可能误杀无关进程。
    # 这里校验 /proc/<pid>/cmdline 确实是本项目的 uvicorn 才终止，否则只清理陈旧 PID 文件。
    if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
      if [[ -r "/proc/$pid/cmdline" ]] && ! tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q 'uvicorn'; then
        warn "$(basename "$f") 中的 PID $pid 不是本项目的 uvicorn 进程，跳过终止"
        warn "  实际命令：$(tr '\0' ' ' < "/proc/$pid/cmdline" | cut -c1-120)"
      else
        info "终止进程 $pid（$(basename "$f")）"
        kill "$pid" 2>/dev/null || true
        sleep 2
        if kill -0 "$pid" >/dev/null 2>&1; then
          kill -9 "$pid" 2>/dev/null || true
        fi
        stopped=$((stopped + 1))
      fi
    fi
    rm -f "$f" 2>/dev/null || true
  done
  if command -v systemctl >/dev/null 2>&1 && systemctl is-active rag-agent >/dev/null 2>&1; then
    info "停止 systemd 单元 rag-agent"
    sudo systemctl stop rag-agent || true
    stopped=$((stopped + 1))
  fi
  if [[ "$stopped" -gt 0 ]]; then ok "服务已停止"; else info "没有发现运行中的服务进程"; fi
}

cmd_logs() {
  tail -n "${LINES:-120}" -f "$SERVICE_LOG"
}

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
  all)
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
    echo "  bash scripts/deploy_server.sh systemd     # 开机自启"
    echo "  bash scripts/deploy_server.sh status      # 状态总览"
    echo "  bash scripts/run_local.sh eval            # 固定问题集评测"
    ;;
  status) cmd_status ;;
  stop)   cmd_stop ;;
  logs)   cmd_logs ;;
  -h|--help|help) cmd_help ;;
  *) echo "未知参数：$1" >&2; cmd_help >&2; exit 1 ;;
esac
