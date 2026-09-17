#!/usr/bin/env bash
# lib.sh —— local.sh / server.sh 共用的函数库
#
# 本文件只定义函数与常量，被 source 后不执行任何动作（不 cd、不建目录、不装 trap）。
# 调用方负责：source 本文件 → 设置自己的默认值 → init_paths → install_err_trap。
#
# 由 scripts/local.sh 与 scripts/server.sh 共同依赖，请勿单独执行。

LIB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$LIB_DIR/.." && pwd)"

# ---------------------------------------------------------------- 输出工具
hr() { printf '%s\n' "------------------------------------------------------------"; }
banner() { echo; echo "======== $* ========"; }
ok()   { echo "   ✅ $*"; }
info() { echo "   ·  $*"; }
warn() { echo "   ⚠️  $*" >&2; }
die()  { echo "   ❌ $*" >&2; exit 1; }

# 步骤计数器：调用方先 step_init <总数>，再逐步 step "标题" … done_step。
# 未声明总数（STEP_TOTAL=1，例如单独执行某个 stage）时只打印 ▶ 标题，
# 不显示无意义的 [1/1] 计数。
STEP_INDEX=0
STEP_TOTAL=1
STEP_START=0
step_init() {
  STEP_INDEX=0
  STEP_TOTAL="${1:-1}"
  STEP_START=0
}
step() {
  STEP_INDEX=$((STEP_INDEX + 1))
  STEP_START=$SECONDS
  echo
  hr
  if (( STEP_TOTAL > 1 )); then
    echo "[$STEP_INDEX/$STEP_TOTAL] $*"
  else
    echo "▶ $*"
  fi
  hr
}
done_step() { echo "   ⏱  耗时 $((SECONDS - STEP_START)) 秒"; }

# ---------------------------------------------------------------- 失败可见性
# set -e 下失败命令直接退出，且很多命令自身不打印可读信息，表现为「跑一半静默回到
# 提示符」。装上 ERR trap，把行号、失败命令、退出码打出来。
# ERR_SEEN_LINE 用于去重：同一条命令触发的重复 ERR 只报一次。调用方可覆盖 ERR_LABEL。
ERR_LABEL="执行中断"
ERR_FILE=""
ERR_SEEN_LINE=""

_report_err() {
  local status="$1" line="$2" cmd="$3"
  [[ "$line" == "${ERR_SEEN_LINE:-}" ]] && return 0
  ERR_SEEN_LINE="$line"
  printf '\n   ❌ %s：第 %s 行执行失败（退出码 %s）\n' "$ERR_LABEL" "$line" "$status" >&2
  printf '      失败命令：%s\n' "$cmd" >&2
  printf '      文件位置：%s\n' "$ERR_FILE" >&2
  printf '      查看该行：sed -n %sp %s\n' "$line" "$ERR_FILE" >&2
}

install_err_trap() {
  ERR_LABEL="${1:-执行中断}"
  ERR_FILE="${2:-${BASH_SOURCE[1]:-$0}}"
  # shellcheck disable=SC2016
  trap 'err_status=$?; _report_err "$err_status" "$LINENO" "$BASH_COMMAND"' ERR
}

require_cmd() {
  local name="$1" hint="${2:-}"
  command -v "$name" >/dev/null 2>&1 && return 0
  if [[ -n "$hint" ]]; then
    die "缺少 $name（$hint）"
  fi
  die "缺少 $name"
}

# ---------------------------------------------------------------- 解释器定位
# 解析环境前缀下的解释器。刻意不使用 `conda run`：它的 --no-capture-output 需要较新
# 的 conda（老版本会报 "unrecognized arguments"），且 conda run 会捕获/缓冲子进程输出，
# 导致安装过程看不到实时进度。直接调用环境自带的解释器最稳，也不依赖环境已激活。
# sed -n '1p' 而非 head -n 1：head 提前退出会让上游收到 SIGPIPE，在 pipefail 下整条
# 管道被判失败。awk 会读完整个输入，同样避免了提前退出。
resolve_env_python() {
  local env_name="$1" prefix candidate
  [[ -n "$env_name" ]] || return 1
  command -v conda >/dev/null 2>&1 || return 1
  prefix="$(conda env list 2>/dev/null \
    | awk -v name="$env_name" '$1 == name { print $NF }' \
    | sed -n '1p' | tr -d '\r' || true)"
  prefix="${prefix//\\//}"
  for candidate in "$prefix/bin/python" "$prefix/python.exe" "$prefix/bin/python3"; do
    if [[ -n "$prefix" && -x "$candidate" ]]; then
      printf '%s' "$candidate"
      return 0
    fi
  done
  return 1
}

# 定位解释器并缓存到 PYTHON_BIN。PYTHON_BIN 显式设置且可执行时直接采用。
locate_python() {
  if [[ -n "${PYTHON_BIN:-}" && -x "${PYTHON_BIN:-}" ]]; then
    printf '%s' "$PYTHON_BIN"
    return 0
  fi
  local resolved=""
  resolved="$(resolve_env_python "${CONDA_ENV_NAME:-}" || true)"
  if [[ -n "$resolved" ]]; then
    PYTHON_BIN="$resolved"
  else
    PYTHON_BIN="${PYTHON_BIN:-python}"
    warn "未定位到 Conda 环境 ${CONDA_ENV_NAME:-（未指定）}，回退到 $PYTHON_BIN（需已激活对应环境）"
  fi
  printf '%s' "$PYTHON_BIN"
}

py() { "$(locate_python)" "$@"; }

# ---------------------------------------------------------------- 依赖检查
# 服务器侧的严格检查：find_spec 不真正 import，避免在缺 CUDA 时被 torch 拖慢；
# REQUIRE_CUDA=1 时额外要求 CUDA 可用；最后 import api 验证项目自身可加载。
env_deps_ok() {
  local py="${1:-$(locate_python)}"
  REQUIRE_CUDA="${REQUIRE_CUDA:-0}" "$py" - <<'PY'
import importlib.util
import os
import sys

required_modules = (
    "fastapi",
    "uvicorn",
    "mcp",
    "langgraph",
    "langchain_chroma",
    "langchain_community",
    "langchain_text_splitters",
    "httpx",
    "torch",
    "modelscope",
    "sentence_transformers",
    "transformers",
    "dotenv",
    "pypdf",
    "multipart",
    "chromadb",
    "streamlit",
)
missing = [name for name in required_modules if importlib.util.find_spec(name) is None]
if missing:
    print("缺少模块: " + ", ".join(missing), file=sys.stderr)
    raise SystemExit(2)

import torch

if os.environ.get("REQUIRE_CUDA", "0") == "1" and not torch.cuda.is_available():
    print("PyTorch 已安装，但当前环境无法访问 CUDA。", file=sys.stderr)
    raise SystemExit(3)

import api  # noqa: F401

print(f"依赖检查通过: Python {sys.version.split()[0]}, torch {torch.__version__}")
PY
}

# 本地侧的自检：真正 import 每个模块，并打印 GPU 信息；缺依赖时列出清单后退出 1。
check_env_imports() {
  local py="${1:-$(locate_python)}"
  "$py" - <<'PY'
import importlib
import sys

required = [
    "torch", "transformers", "sentence_transformers", "chromadb",
    "langgraph", "langchain_chroma", "langchain_community", "mcp",
    "fastapi", "uvicorn", "pypdf", "streamlit", "modelscope", "httpx", "multipart",
]
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception:
        missing.append(name)
if missing:
    print("   - 缺少依赖：" + ", ".join(missing))
    sys.exit(1)
print(f"   - 依赖齐全（{len(required)} 项）")
try:
    import torch

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"   - GPU: {torch.cuda.get_device_name(0)} / {props.total_memory / 1024 ** 3:.1f} GB")
    else:
        print("   - GPU: 不可用，将走 CPU（.env 中 DEVICE=auto 会自动适配）")
except Exception as exc:  # pragma: no cover
    print(f"   - GPU 检测跳过：{type(exc).__name__}")
PY
}

# 三个模型是否已缓存到 MODEL_CACHE_DIR，只打印不判定。
print_model_cache_status() {
  local py="${1:-$(locate_python)}"
  "$py" - <<'PY'
from agentic_rag import Settings

settings = Settings.from_env()
cached = []
for model_id in (settings.embedding_model, settings.reranker_model, settings.llm_model):
    organization, _, name = model_id.partition("/")
    hits = [
        settings.model_cache_dir / organization / name,
        settings.model_cache_dir / organization / name.replace(".", "___"),
    ]
    cached.append(f"{model_id}={ '已缓存' if any(p.exists() for p in hits) else '未缓存' }")
print("   - " + " | ".join(cached))
print("   - 首次 build 会从 ModelScope 下载缺失模型（BGE-M3 约 2.2GB、Reranker 约 1.1GB、Qwen2.5-1.5B 约 3.1GB）")
PY
}

# 打印知识库内每个文档的状态，只打印不判定。
print_document_records() {
  local py="${1:-$(locate_python)}"
  "$py" - <<'PY'
from agentic_rag import Settings
from agentic_rag.knowledge_base import KnowledgeBase
from agentic_rag.models import ModelRuntime

settings = Settings.from_env()
kb = KnowledgeBase(settings, ModelRuntime(settings))
records = kb.document_records()
if not records:
    print("   - 目录为空：需要先建库")
for record in records:
    print(
        f"   - {record.source}: {record.status} / {record.chunk_count} 块 / "
        f"version={(record.active_version_id or '-')[:12]}"
    )
PY
}

# ---------------------------------------------------------------- .env 读写
env_get() {
  local key="$1"
  [[ -f .env ]] || { printf ''; return 0; }
  sed -n "s/^${key}=//p" .env | tail -n 1 | tr -d '\r'
}

# 幂等写入：已存在同名键则原地替换，否则追加。转义替换串里的 \ & |，
# 否则含这些字符的值（如 MCP_ALLOWED_HOSTS 里的 [::1]、域名通配）会破坏 sed 表达式。
env_set() {
  local key="$1" value="$2" escaped
  [[ -f .env ]] || cp .env.example .env
  escaped="$(printf '%s' "$value" | sed -e 's/[\\&|]/\\&/g')"
  if grep -q "^${key}=" .env; then
    sed -i "s|^${key}=.*|${key}=${escaped}|" .env
  else
    printf '\n%s=%s\n' "$key" "$value" >> .env
  fi
}

api_key() { env_get API_KEY; }

ensure_env_file() {
  if [[ -f .env ]]; then
    info ".env 已存在，保持不变"
  else
    cp .env.example .env
    ok "已从 .env.example 生成 .env"
  fi
}

# openssl 优先（无解释器依赖），否则退回 python 的 secrets。
generate_api_key() {
  if command -v openssl >/dev/null 2>&1; then
    openssl rand -hex 32
    return 0
  fi
  local py_bin=""
  if command -v python3 >/dev/null 2>&1; then
    py_bin="python3"
  else
    py_bin="$(locate_python 2>/dev/null || true)"
  fi
  [[ -n "$py_bin" ]] || return 1
  "$py_bin" -c 'import secrets; print(secrets.token_hex(32))'
}

ensure_api_key() {
  ensure_env_file
  local key generated
  key="$(env_get API_KEY)"
  if [[ -n "$key" ]]; then
    info ".env 中已有 API_KEY"
  else
    generated="$(generate_api_key | tr -d '\r\n')" || true
    [[ -n "$generated" ]] || die "无法生成 API Key，需要 openssl 或 python3"
    env_set API_KEY "$generated"
    ok "已生成随机 API_KEY 并写回 .env"
  fi
  chmod 600 .env 2>/dev/null || true
}

# ---------------------------------------------------------------- 路径与常量
# 必须在调用方设好 APP_PORT 之后再调用。
init_paths() {
  RUN_DIR="${RUN_DIR:-runs}"
  PID_FILE="${PID_FILE:-$RUN_DIR/service.pid}"
  LEGACY_PID_FILE="${LEGACY_PID_FILE:-$RUN_DIR/uvicorn.pid}"
  SERVICE_LOG="${SERVICE_LOG:-$RUN_DIR/service.log}"
  BODY_FILE="${BODY_FILE:-$RUN_DIR/.last_body.json}"
  BASE_URL="${BASE_URL:-http://127.0.0.1:${APP_PORT:-8080}}"
  READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-240}"
  mkdir -p "$RUN_DIR"
}

# ---------------------------------------------------------------- HTTP 助手
# 只取状态码、丢弃响应体。curl 连接失败时仍会通过 -w 输出 000，同时又以非 0 退出，
# 所以这里用 `|| true` 吞掉退出码（否则 set -e 会中断脚本），而不能写成 `|| echo 000`
# ——后者会把 curl 自己输出的 000 再拼一个，得到 "000000"。
probe_status() {
  curl -sS -o /dev/null -w '%{http_code}' "$@" 2>/dev/null || true
}

# 状态码写 stdout，响应体落到 $BODY_FILE；连接失败统一返回 "000" 而非中断脚本。
http_get() {
  curl -sS -o "$BODY_FILE" -w '%{http_code}' \
    -H "X-API-Key: $(api_key)" "$BASE_URL$1" 2>/dev/null || echo "000"
}

http_post() {
  curl -sS -o "$BODY_FILE" -w '%{http_code}' -X POST \
    -H "X-API-Key: $(api_key)" -H 'Content-Type: application/json' \
    -d "$2" "$BASE_URL$1" 2>/dev/null || echo "000"
}

json_get() {
  "$(locate_python)" -c "
import json, sys
try:
    data = json.load(open(sys.argv[1], encoding='utf-8'))
except Exception as exc:
    print(f'<JSON 解析失败: {type(exc).__name__}>'); raise SystemExit
node = data
for key in sys.argv[2:]:
    if isinstance(node, list):
        node = node[int(key)]
    else:
        node = node.get(key)
    if node is None:
        print(''); raise SystemExit
print(node if not isinstance(node, (dict, list)) else json.dumps(node, ensure_ascii=False)[:400])
" "$BODY_FILE" "$@"
}

# 从评测集里挑一个「有期望来源」的问题，用于在线验收。
probe_question() {
  "$(locate_python)" - <<'PY'
import json
from pathlib import Path

path = Path("eval/dataset.example.jsonl")
fallback = "知识库里有哪些文档？"
if not path.exists():
    print(fallback)
else:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("expected_sources"):
            print(item["question"])
            break
    else:
        print(fallback)
PY
}

# ---------------------------------------------------------------- 服务托管
pid_alive() { kill -0 "$1" 2>/dev/null; }

service_pid() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid="$(tr -d ' \r\n' < "$PID_FILE")"
  [[ -n "$pid" ]] || return 1
  pid_alive "$pid" || return 1
  printf '%s' "$pid"
}

# 兼容旧版脚本写下的 runs/uvicorn.pid，避免遗留进程无法被 stop/status 看到。
legacy_service_pid() {
  [[ -f "$LEGACY_PID_FILE" ]] || return 1
  local pid
  pid="$(tr -d ' \r\n' < "$LEGACY_PID_FILE")"
  [[ -n "$pid" ]] || return 1
  pid_alive "$pid" || return 1
  printf '%s' "$pid"
}

service_alive() {
  [[ "$(probe_status "$BASE_URL/v1/health/live")" == "200" ]]
}

# 仅在确认是本项目的 uvicorn 进程时才终止：Linux 下先读 /proc/<pid>/cmdline 核对身份，
# 避免 PID 被复用后误伤无关进程。
terminate_pid() {
  local pid="$1" cmdline=""
  pid_alive "$pid" || return 0
  if [[ -r "/proc/$pid/cmdline" ]]; then
    cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
  fi
  if [[ -n "$cmdline" && "$cmdline" != *uvicorn* ]]; then
    warn "PID $pid 不是 uvicorn 进程，跳过终止（可能是 PID 复用）"
    return 0
  fi
  kill "$pid" 2>/dev/null || true
  sleep 1
  if pid_alive "$pid"; then
    if command -v taskkill >/dev/null 2>&1; then
      taskkill //F //T //PID "$pid" >/dev/null 2>&1 || true
    fi
    pid_alive "$pid" && kill -9 "$pid" 2>/dev/null || true
  fi
}

stop_service() {
  local pid
  if pid="$(service_pid)"; then
    info "终止服务进程 $pid"
    terminate_pid "$pid"
  fi
  if pid="$(legacy_service_pid)"; then
    info "发现旧版 PID 文件 $LEGACY_PID_FILE（PID $pid），一并清理"
    terminate_pid "$pid"
  fi
  rm -f "$PID_FILE" "$LEGACY_PID_FILE"
  ok "服务已停止"
}

start_service() {
  local pid py
  if service_alive; then
    info "已有实例在 $BASE_URL 上响应，直接复用"
    return 0
  fi
  if pid="$(service_pid)"; then
    warn "发现残留 PID $pid 但服务未就绪，先终止它"
    stop_service || true
  fi
  py="$(locate_python)"
  info "后台启动：$py -m uvicorn api:app --host 127.0.0.1 --port $APP_PORT"
  info "日志：$SERVICE_LOG"
  ( nohup "$py" -m uvicorn api:app \
      --host 127.0.0.1 --port "$APP_PORT" > "$SERVICE_LOG" 2>&1 &
    echo $! > "$PID_FILE" )
  local waited=0
  while (( waited < READY_TIMEOUT_SECONDS )); do
    if service_alive; then
      ok "服务已就绪（等待 ${waited}s）"
      return 0
    fi
    if ! service_pid >/dev/null; then
      echo "--- $SERVICE_LOG 末尾 ---" >&2
      tail -20 "$SERVICE_LOG" >&2 || true
      die "服务进程已退出，启动失败"
    fi
    sleep 2
    waited=$((waited + 2))
  done
  tail -20 "$SERVICE_LOG" >&2 || true
  die "等待服务就绪超时（${READY_TIMEOUT_SECONDS}s）"
}

# ---------------------------------------------------------------- 离线验证
run_verify() {
  local py
  py="$(locate_python)"
  echo "== 项目目录：$PROJECT_DIR"
  echo "== 解释器：$py ($("$py" -c 'import sys;print(sys.version.split()[0])'))"

  echo
  echo "== [1/3] 语法检查"
  "$py" -m compileall -q agentic_rag api.py tests scripts/offline_smoke.py
  echo "   compileall 通过"

  echo
  echo "== [2/3] 自动化测试"
  "$py" -m pytest -q -p no:cacheprovider

  echo
  echo "== [3/3] 离线端到端冒烟（不下载模型）"
  "$py" scripts/offline_smoke.py

  echo
  echo "全部验证通过。"
}

# ---------------------------------------------------------------- 在线验收
# 8 项：存活 / 就绪 / 文档清单 / 版本查询 / 问答 / 版本回退 / 指标 / 鉴权边界。
# 回退接口允许 409（只有一个健康版本时属预期）；无 Key 访问返回非 401 只 warn，
# 因为 .env 未设置 API_KEY 时门禁按设计关闭。
online_acceptance() {
  local key code code401
  key="$(api_key)"
  [[ -n "$key" ]] || die ".env 中没有 API_KEY，先执行 env 子命令生成"

  info "1) 存活探针 /v1/health/live"
  [[ "$(probe_status "$BASE_URL/v1/health/live")" == "200" ]] \
    || die "/v1/health/live 未返回 200"
  ok "存活探针通过"

  info "2) 就绪探针 /v1/health/ready"
  code="$(http_get /v1/health/ready)"
  [[ "$code" == "200" ]] || { cat "$BODY_FILE"; die "/v1/health/ready 返回 $code"; }
  ok "就绪（catalog.integrity=$(json_get catalog integrity)，llm_backend=$(json_get llm_backend)，orchestration=$(json_get orchestration)）"
  info "   MCP 工具：$(json_get mcp_tools)"

  info "3) 文档清单 /v1/documents"
  code="$(http_get /v1/documents)"
  [[ "$code" == "200" ]] || die "/v1/documents 返回 $code"
  ok "已索引文档：$(json_get documents | head -c 300)"

  info "4) 版本查询 /v1/documents/versions"
  local doc_source versions_count
  doc_source="$(json_get documents 0 source)"
  versions_count=0
  if [[ -n "$doc_source" ]]; then
    code="$(http_get "/v1/documents/versions?source=$(printf '%s' "$doc_source" | sed 's/ /%20/g')")"
    [[ "$code" == "200" ]] || die "/v1/documents/versions 返回 $code"
    # 立刻取出数量：后面的请求会覆盖 $BODY_FILE。
    versions_count="$("$(locate_python)" -c "
import json
doc = json.load(open('$BODY_FILE', encoding='utf-8'))
print(len(doc.get('versions', [])) if isinstance(doc, dict) else 0)
")"
    ok "共 $versions_count 个版本：$(json_get versions | head -c 300)"
  else
    warn "知识库为空，跳过版本与回退演示"
  fi

  local payload
  payload="$("$(locate_python)" -c "
import json, sys
print(json.dumps({'question': sys.argv[1], 'history': []}, ensure_ascii=False))
" "$(probe_question)")"

  if [[ "${SKIP_ONLINE_QUERY:-0}" == "1" ]]; then
    info "5) 问答 / 6) 版本回退：SKIP_ONLINE_QUERY=1，跳过（不加载模型）"
  else
    info "5) 问答 /v1/query（真实模型推理）"
    info "   提问：$(probe_question)"
    code="$(http_post /v1/query "$payload")"
    [[ "$code" == "200" ]] || { cat "$BODY_FILE"; die "/v1/query 返回 $code"; }
    ok "run_id=$(json_get run_id) | grounded=$(json_get grounded) | 时延=$(json_get latency_ms) ms"
    ok "回答：$(json_get answer)"
    ok "引用证据：$(json_get evidence | head -c 300)"
    ok "编排轨迹：$(json_get trace | head -c 400)"

    info "6) 版本回退 /v1/documents/rollback"
    if [[ -n "$doc_source" ]]; then
      local rb_payload
      rb_payload="$("$(locate_python)" -c "
import json, sys
print(json.dumps({'source': sys.argv[1]}, ensure_ascii=False))
" "$doc_source")"
      code="$(http_post /v1/documents/rollback "$rb_payload")"
      if [[ "$code" == "200" ]]; then
        ok "已回退到版本 $(json_get active_version_id)（回退前共 $versions_count 个版本）"
      elif [[ "$code" == "409" ]]; then
        info "返回 409：$(json_get detail)"
        info "   ↳ 预期行为：该文档只有 1 个健康版本，没有可回退的历史版本（共 $versions_count 个版本）。"
      else
        warn "回退接口返回 $code：$(cat "$BODY_FILE")"
      fi
    fi
  fi

  info "7) Prometheus 指标 /metrics"
  code="$(http_get /metrics)"
  [[ "$code" == "200" ]] || die "/metrics 返回 $code"
  ok "指标样例：$(grep -m1 '^agentic_rag_http_requests_total' "$BODY_FILE" || echo '(尚无语义指标，先发一次请求)')"

  info "8) 鉴权边界：无 Key 访问 /v1/query 应被拒绝"
  code401="$(probe_status -X POST -H 'Content-Type: application/json' -d "$payload" "$BASE_URL/v1/query")"
  if [[ "$code401" == "401" ]]; then
    ok "无 Key 访问返回 401，API Key 门禁生效"
  else
    warn "无 Key 访问返回 $code401（若 .env 未设置 API_KEY，则门禁按设计关闭）"
  fi
}
