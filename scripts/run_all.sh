#!/usr/bin/env bash
# run_all.sh —— 从零开始，一条命令把整个项目跑完
#
# 覆盖：环境自检 → 生成 .env/API Key → 准备 data/ → 离线自检(pytest+冒烟)
#       → 下载模型并建向量库 → 固定问题集评测 → 后台起服务 → 在线接口验收 → 汇总
#
# 用法：
#   bash scripts/run_all.sh              # 全流程（首次约需下载 6.4GB 模型）
#   bash scripts/run_all.sh stop         # 停止后台服务
#   bash scripts/run_all.sh status       # 查看服务与知识库状态
#
# 可选开关（前置赋值即可）：
#   SKIP_VERIFY=1   跳过 scripts/verify.sh
#   SKIP_BUILD=1    跳过模型下载与建库
#   SKIP_EVAL=1     跳过固定问题集评测
#   SKIP_SERVE=1    跳过起服务与在线验收
#   SKIP_ONLINE_QUERY=1  在线验收时不发真实问答（不加载模型，只验接口与鉴权）
#   CONDA_ENV_NAME / PYTHON_BIN / APP_PORT / READY_TIMEOUT_SECONDS 可覆盖默认值
#
# 注意：本脚本不创建 Conda 环境、不安装依赖。环境创建请见 README「快速开始（Conda）」。
set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-ai_project}"
APP_PORT="${APP_PORT:-8080}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-240}"
BASE_URL="http://127.0.0.1:${APP_PORT}"

SKIP_VERIFY="${SKIP_VERIFY:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_SERVE="${SKIP_SERVE:-0}"
SKIP_ONLINE_QUERY="${SKIP_ONLINE_QUERY:-0}"

RUN_DIR="runs"
PID_FILE="$RUN_DIR/service.pid"
SERVICE_LOG="$RUN_DIR/service.log"
BODY_FILE="$RUN_DIR/.last_body.json"
mkdir -p "$RUN_DIR"

STEP_INDEX=0
STEP_TOTAL=7
STEP_START=0

# ---------------------------------------------------------------- 输出工具
hr() { printf '%s\n' "------------------------------------------------------------"; }
step() {
  STEP_INDEX=$((STEP_INDEX + 1))
  STEP_START=$SECONDS
  echo
  hr
  echo "[$STEP_INDEX/$STEP_TOTAL] $*"
  hr
}
ok()   { echo "   ✅ $*"; }
info() { echo "   ·  $*"; }
warn() { echo "   ⚠️  $*"; }
die()  { echo "   ❌ $*" >&2; exit 1; }
done_step() { echo "   ⏱  耗时 $((SECONDS - STEP_START)) 秒"; }

# ---------------------------------------------------------------- 解释器定位
if [[ -z "${PYTHON_BIN:-}" ]] && command -v conda >/dev/null 2>&1; then
  ENV_PREFIX="$(conda env list 2>/dev/null \
    | sed -n "s/^${CONDA_ENV_NAME}[[:space:]][[:space:]]*//p" \
    | head -n 1 | tr -d '\r')"
  ENV_PREFIX="${ENV_PREFIX//\\//}"
  for candidate in "$ENV_PREFIX/python.exe" "$ENV_PREFIX/bin/python"; do
    if [[ -n "$ENV_PREFIX" && -x "$candidate" ]]; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi
[[ -n "${PYTHON_BIN:-}" && -x "$PYTHON_BIN" ]] || PYTHON_BIN="python"
export PYTHON_BIN

# ==== 读取 .env 中的 API Key（缺失时为空串）====
api_key() {
  [[ -f .env ]] || { printf ''; return 0; }
  sed -n 's/^API_KEY=//p' .env | tail -n 1 | tr -d '\r'
}

# ==== 从评测集里挑一个「有期望来源」的问题，用于在线验收 ====
probe_question() {
  "$PYTHON_BIN" - <<'PY'
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

# ==== HTTP 助手：返回状态码，响应体写入 $BODY_FILE ====
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
  "$PYTHON_BIN" -c "
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

# ---------------------------------------------------------------- 服务管理
service_pid() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid="$(tr -d ' \r\n' < "$PID_FILE")"
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  printf '%s' "$pid"
}

service_alive() {
  [[ "$(curl -sS -o /dev/null -w '%{http_code}' "$BASE_URL/v1/health/live" 2>/dev/null || echo 000)" == "200" ]]
}

start_service() {
  if service_alive; then
    info "已有实例在 $BASE_URL 上响应，直接复用"
    return 0
  fi
  if pid="$(service_pid)"; then
    warn "发现残留 PID $pid 但服务未就绪，先终止它"
    stop_service || true
  fi
  info "后台启动：$PYTHON_BIN -m uvicorn api:app --host 127.0.0.1 --port $APP_PORT"
  info "日志：$SERVICE_LOG"
  ( nohup "$PYTHON_BIN" -m uvicorn api:app \
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

stop_service() {
  if pid="$(service_pid)"; then
    info "终止服务进程 $pid"
    kill "$pid" 2>/dev/null || true
    sleep 1
    if kill -0 "$pid" 2>/dev/null; then
      command -v taskkill >/dev/null 2>&1 && taskkill //F //T //PID "$pid" >/dev/null 2>&1 || kill -9 "$pid" 2>/dev/null || true
    fi
  fi
  rm -f "$PID_FILE"
  ok "服务已停止"
}

cmd_status() {
  echo "项目目录：$PROJECT_DIR"
  echo "解释器：  $PYTHON_BIN"
  echo "服务地址：$BASE_URL"
  if service_alive; then
    echo "服务状态：运行中（HTTP 200）"
    echo "API Key： $(api_key)"
  else
    echo "服务状态：未运行"
  fi
  echo
  "$PYTHON_BIN" - <<'PY'
from agentic_rag import Settings
from agentic_rag.knowledge_base import KnowledgeBase
from agentic_rag.models import ModelRuntime

settings = Settings.from_env()
kb = KnowledgeBase(settings, ModelRuntime(settings))
records = kb.document_records()
if not records:
    print("知识库：空（先执行 bash scripts/run_local.sh build）")
for record in records:
    print(f"知识库：{record.source} | {record.status} | {record.chunk_count} 块 | version={(record.active_version_id or '-')[:12]}")
PY
}

# ---------------------------------------------------------------- 在线验收
online_acceptance() {
  local key
  key="$(api_key)"
  [[ -n "$key" ]] || die ".env 中没有 API_KEY，先执行 bash scripts/run_local.sh env"

  info "1) 存活探针 /v1/health/live"
  [[ "$(curl -sS -o /dev/null -w '%{http_code}' "$BASE_URL/v1/health/live")" == "200" ]] \
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
  SOURCE="$("$PYTHON_BIN" -c "
import json
docs = json.load(open('$BODY_FILE', encoding='utf-8'))['documents']
print(docs[0]['source'] if docs else '')
")"
  VERSIONS_COUNT=0
  if [[ -n "$SOURCE" ]]; then
    code="$(http_get "/v1/documents/versions?source=$(printf '%s' "$SOURCE" | sed 's/ /%20/g')")"
    [[ "$code" == "200" ]] || die "/v1/documents/versions 返回 $code"
    # 立即取出数量，避免后续请求覆盖 BODY_FILE。
    VERSIONS_COUNT="$("$PYTHON_BIN" -c "
import json
doc = json.load(open('$BODY_FILE', encoding='utf-8'))
print(len(doc.get('versions', [])) if isinstance(doc, dict) else 0)
")"
    ok "共 $VERSIONS_COUNT 个版本：$(json_get versions | head -c 300)"
  else
    warn "知识库为空，跳过版本与回退演示"
  fi

  local payload
  payload="$("$PYTHON_BIN" -c "
import json, sys
print(json.dumps({'question': sys.argv[1], 'history': []}, ensure_ascii=False))
" "$(probe_question)")"

  if [[ "$SKIP_ONLINE_QUERY" == "1" ]]; then
    info "5) 问答 / 6) 版本回退：SKIP_ONLINE_QUERY=1，跳过（不加载模型）"
  else
    info "5) 问答 /v1/query（真实模型推理）"
    QUESTION="$(probe_question)"
    info "   提问：$QUESTION"
    code="$(http_post /v1/query "$payload")"
    [[ "$code" == "200" ]] || { cat "$BODY_FILE"; die "/v1/query 返回 $code"; }
    ok "run_id=$(json_get run_id) | grounded=$(json_get grounded) | 时延=$(json_get latency_ms) ms"
    ok "回答：$(json_get answer)"
    ok "引用证据：$(json_get evidence | head -c 300)"
    ok "编排轨迹：$(json_get trace | head -c 400)"

    info "6) 版本回退 /v1/documents/rollback"
    if [[ -n "$SOURCE" ]]; then
      local rb_payload
      rb_payload="$("$PYTHON_BIN" -c "
import json, sys
print(json.dumps({'source': sys.argv[1]}, ensure_ascii=False))
" "$SOURCE")"
      code="$(http_post /v1/documents/rollback "$rb_payload")"
      if [[ "$code" == "200" ]]; then
        ok "已回退到版本 $(json_get active_version_id)（回退前共 $VERSIONS_COUNT 个版本）"
      elif [[ "$code" == "409" ]]; then
        info "返回 409：$(json_get detail)"
        info "   ↳ 预期行为：该文档只有 1 个健康版本，没有可回退的历史版本（共 $VERSIONS_COUNT 个版本）。"
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
  local code401
  code401="$(curl -sS -o /dev/null -w '%{http_code}' -X POST \
    -H 'Content-Type: application/json' -d "$payload" "$BASE_URL/v1/query" 2>/dev/null || echo 000)"
  if [[ "$code401" == "401" ]]; then
    ok "无 Key 访问返回 401，API Key 门禁生效"
  else
    warn "无 Key 访问返回 $code401（若 .env 未设置 API_KEY，则门禁按设计关闭）"
  fi
}

# ---------------------------------------------------------------- 主流程
main() {
  echo "############################################################"
  echo "# RAG 知识库问答 Agent —— 从零全流程"
  echo "# 项目：$PROJECT_DIR"
  echo "# 解释器：$PYTHON_BIN"
  echo "# 服务地址：$BASE_URL"
  echo "############################################################"

  # 1. 环境与依赖
  step "环境与依赖自检"
  command -v curl >/dev/null 2>&1 || die "缺少 curl，无法做在线验收"
  bash scripts/run_local.sh check
  ok "环境自检通过"
  ok "解释器：$PYTHON_BIN（$("$PYTHON_BIN" -c 'import sys;print(sys.version.split()[0])')）"
  done_step

  # 2. 配置
  step "生成 .env 与 API Key"
  bash scripts/run_local.sh env
  ok "API Key 前 8 位：$(api_key | cut -c1-8)…"
  done_step

  # 3. 数据
  step "准备知识库源文件 data/"
  bash scripts/run_local.sh data
  ok "PDF 数量：$( { find data -maxdepth 1 -iname '*.pdf' || true; } | wc -l | tr -d ' ')"
  done_step

  # 4. 离线自检
  step "离线自检（语法 + pytest + 不下载模型的端到端冒烟）"
  if [[ "$SKIP_VERIFY" == "1" ]]; then
    warn "SKIP_VERIFY=1，跳过"
  else
    bash scripts/verify.sh
    ok "离线自检全部通过"
  fi
  done_step

  # 5. 建库
  step "下载模型并构建向量知识库"
  if [[ "$SKIP_BUILD" == "1" ]]; then
    warn "SKIP_BUILD=1，跳过"
  else
    info "首次运行需要从 ModelScope 下载 BGE-M3 / BGE Reranker / Qwen2.5-1.5B（合计约 6.4GB）"
    bash scripts/run_local.sh build
    ok "向量库构建完成"
    "$PYTHON_BIN" - <<'PY'
from agentic_rag import Settings
from agentic_rag.knowledge_base import KnowledgeBase
from agentic_rag.models import ModelRuntime

settings = Settings.from_env()
kb = KnowledgeBase(settings, ModelRuntime(settings))
for record in kb.document_records():
    print(f"   ·  {record.source}: {record.status} / {record.chunk_count} 块")
PY
  fi
  done_step

  # 6. 评测
  step "固定问题集评测（检索命中率 / 引用准确性）"
  if [[ "$SKIP_EVAL" == "1" ]]; then
    warn "SKIP_EVAL=1，跳过"
  else
    bash scripts/run_local.sh eval
    ok "报告：$(ls -t reports/*.json 2>/dev/null | head -n 1 || echo '（见 reports/）')"
    ok "可读版：$(ls -t reports/*.md 2>/dev/null | head -n 1 || echo '（见 reports/）')"
  fi
  done_step

  # 7. 服务与在线验收
  step "后台启动 FastAPI 并做在线接口验收"
  if [[ "$SKIP_SERVE" == "1" ]]; then
    warn "SKIP_SERVE=1，跳过"
  else
    start_service
    online_acceptance
  fi
  done_step

  echo
  hr
  echo "全部完成"
  hr
  echo "服务地址：$BASE_URL"
  echo "OpenAPI： $BASE_URL/docs"
  echo "MCP：     $BASE_URL/mcp/  （可用工具：search_knowledge_base / query_knowledge_base / list_knowledge_documents / knowledge_base_health）"
  echo "API Key： $(api_key)"
  echo "服务日志：$SERVICE_LOG"
  echo "停止服务：bash scripts/run_all.sh stop"
  echo "查看状态：bash scripts/run_all.sh status"
  echo "评测报告：reports/"
  echo "运行日志：runs/agent_runs.jsonl（含 run_id 与每步 record_id/evidence）"
  echo
  echo "若只想本地看界面：bash scripts/run_local.sh ui  →  $BASE_URL 之外的 http://localhost:8501"
}

case "${1:-run}" in
  run)    main ;;
  stop)   stop_service ;;
  status) cmd_status ;;
  -h|--help|help) sed -n '2,22p' "${BASH_SOURCE[0]}" ;;
  *) echo "未知参数：$1（可用：run / stop / status / help）" >&2; exit 1 ;;
esac
