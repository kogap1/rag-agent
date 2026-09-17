#!/usr/bin/env bash
# local.sh —— 本地一条龙跑通知识库问答 Agent（只复用已有 Conda 环境，不创建环境、不装依赖）
#
# 用法：
#   bash scripts/local.sh check      # 环境与数据自检（先跑这个）
#   bash scripts/local.sh env        # 生成 .env 与 API Key
#   bash scripts/local.sh data       # 准备 data/ 下的 PDF（可用 SAMPLE_PDF 指定样例）
#   bash scripts/local.sh build      # 建/重建向量知识库（首次会下载模型）
#   bash scripts/local.sh api        # 前台启动 FastAPI（默认 127.0.0.1:8080）
#   bash scripts/local.sh ui         # 前台启动 Streamlit 界面（默认 8501）
#   bash scripts/local.sh demo       # check + env + data + build + 离线冒烟
#   bash scripts/local.sh eval       # 用固定问题集跑 Agent 评测
#   bash scripts/local.sh verify     # 语法检查 + 全量测试 + 离线冒烟
#   bash scripts/local.sh run        # 一条龙：自检→配置→数据→离线自检→建库→评测→起服务→在线验收
#   bash scripts/local.sh stop       # 停止后台服务
#   bash scripts/local.sh status     # 查看服务与知识库状态
#
# run 的可用开关（前置赋值即可）：
#   SKIP_VERIFY=1 / SKIP_BUILD=1 / SKIP_EVAL=1 / SKIP_SERVE=1
#   SKIP_ONLINE_QUERY=1           在线验收时不发真实问答（不加载模型，只验接口与鉴权）
#
# 其他可覆盖变量：CONDA_ENV_NAME / PYTHON_BIN / APP_PORT / READY_TIMEOUT_SECONDS / SAMPLE_PDF
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"
cd "$PROJECT_DIR"
install_err_trap "本地流程中断"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-ai_project}"
APP_PORT="${APP_PORT:-8080}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-240}"
# 可选：设置 SAMPLE_PDF 指向一个本地 PDF，data/ 为空时会自动复制进去。
SAMPLE_PDF="${SAMPLE_PDF:-}"

SKIP_VERIFY="${SKIP_VERIFY:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_SERVE="${SKIP_SERVE:-0}"

DATA_DIR="${DATA_DIR:-data}"
init_paths

# ---------------------------------------------------------------- env
cmd_env() {
  banner "准备 .env 与 API Key"
  ensure_api_key
}

# ---------------------------------------------------------------- data
cmd_data() {
  banner "准备知识库源文件 $DATA_DIR/"
  mkdir -p "$DATA_DIR"
  local count
  count="$( { find "$DATA_DIR" -maxdepth 1 -iname '*.pdf' 2>/dev/null || true; } | wc -l | tr -d ' ')"
  if [[ "$count" -gt 0 ]]; then
    info "$DATA_DIR/ 已有 $count 个 PDF，直接使用"
    return 0
  fi
  if [[ -n "$SAMPLE_PDF" && -f "$SAMPLE_PDF" ]]; then
    cp "$SAMPLE_PDF" "$DATA_DIR/"
    ok "$DATA_DIR/ 为空，已从样例复制：$(basename "$SAMPLE_PDF")"
  else
    warn "$DATA_DIR/ 为空，且未设置 SAMPLE_PDF 样例。"
    warn "请把自己的 PDF 放进 $PROJECT_DIR/$DATA_DIR/ 后重新执行。"
    return 1
  fi
}

# ---------------------------------------------------------------- check
cmd_check() {
  # data/ 可能还没建；先建出来，避免 find 在 pipefail 下中断脚本。
  mkdir -p "$DATA_DIR"
  local py_bin
  py_bin="$(locate_python)"

  banner "1. 解释器与依赖"
  echo "   - Python: $("$py_bin" -c 'import sys;print(sys.version.split()[0])') @ $py_bin"
  check_env_imports "$py_bin"

  banner "2. 配置与数据"
  if [[ -f .env ]]; then
    echo "   - .env: 存在"
  else
    echo "   - .env: 缺失（执行 bash scripts/local.sh env 生成）"
  fi
  local pdfs
  pdfs="$( { find "$DATA_DIR" -maxdepth 1 -iname '*.pdf' 2>/dev/null || true; } | wc -l | tr -d ' ')"
  echo "   - $DATA_DIR/ 下 PDF: ${pdfs:-0} 个"

  banner "3. 知识库状态"
  print_document_records "$py_bin"

  banner "4. 模型缓存"
  print_model_cache_status "$py_bin"

  banner "下一步"
  echo "   建库：bash scripts/local.sh build"
  echo "   起服务：bash scripts/local.sh api      （OpenAPI: http://127.0.0.1:${APP_PORT}/docs）"
  echo "   起界面：bash scripts/local.sh ui       （http://localhost:8501）"
  echo "   一条龙：bash scripts/local.sh run"
}

# ---------------------------------------------------------------- build
build_db() {
  banner "构建向量知识库（--replace 重建）"
  py create_db.py "$DATA_DIR" --replace
}

cmd_build() {
  cmd_env
  cmd_data
  build_db
}

# ---------------------------------------------------------------- api / ui
# 刻意用 exec 前台运行：Ctrl-C 即可停止，不写 PID，不与 run 的后台托管混用。
cmd_api() {
  cmd_env
  local py_bin key
  py_bin="$(locate_python)"
  key="$(api_key)"
  banner "启动 FastAPI：http://127.0.0.1:${APP_PORT}"
  echo "   - OpenAPI: http://127.0.0.1:${APP_PORT}/docs"
  echo "   - 健康检查: curl -H \"X-API-Key: ${key:0:8}...\" http://127.0.0.1:${APP_PORT}/v1/health/ready"
  echo "   - MCP: http://127.0.0.1:${APP_PORT}/mcp/"
  exec "$py_bin" -m uvicorn api:app --host 127.0.0.1 --port "$APP_PORT"
}

cmd_ui() {
  cmd_env
  banner "启动 Streamlit：http://localhost:8501"
  exec "$(locate_python)" -m streamlit run app.py
}

# ---------------------------------------------------------------- eval
cmd_eval() {
  cmd_env
  banner "固定问题集评测（Agent 模式）"
  py evaluate.py --mode agent --repeats 1
}

# ---------------------------------------------------------------- verify
cmd_verify() { run_verify; }

# ---------------------------------------------------------------- demo
cmd_demo() {
  cmd_check
  cmd_build
  banner "构建后自检"
  py scripts/offline_smoke.py
  banner "完成"
  echo "   接着执行：bash scripts/local.sh api  或  bash scripts/local.sh ui"
}

# ---------------------------------------------------------------- run（一条龙）
cmd_run() {
  step_init 7
  echo "############################################################"
  echo "# RAG 知识库问答 Agent —— 从零全流程"
  echo "# 项目：$PROJECT_DIR"
  echo "# 解释器：$(locate_python)"
  echo "# 服务地址：$BASE_URL"
  echo "############################################################"

  step "环境与依赖自检"
  require_cmd curl "无法做在线验收"
  cmd_check
  ok "环境自检通过"
  done_step

  step "生成 .env 与 API Key"
  cmd_env
  ok "API Key 前 8 位：$(api_key | cut -c1-8)…"
  done_step

  step "准备知识库源文件 $DATA_DIR/"
  cmd_data
  ok "PDF 数量：$( { find "$DATA_DIR" -maxdepth 1 -iname '*.pdf' 2>/dev/null || true; } | wc -l | tr -d ' ')"
  done_step

  step "离线自检（语法 + pytest + 不下载模型的端到端冒烟）"
  if [[ "$SKIP_VERIFY" == "1" ]]; then
    warn "SKIP_VERIFY=1，跳过"
  else
    run_verify
    ok "离线自检全部通过"
  fi
  done_step

  step "下载模型并构建向量知识库"
  if [[ "$SKIP_BUILD" == "1" ]]; then
    warn "SKIP_BUILD=1，跳过"
  else
    info "首次运行需要从 ModelScope 下载 BGE-M3 / BGE Reranker / LLM 生成模型"
    build_db
    print_document_records
    ok "向量库构建完成"
  fi
  done_step

  step "固定问题集评测（检索命中率 / 引用准确性）"
  if [[ "$SKIP_EVAL" == "1" ]]; then
    warn "SKIP_EVAL=1，跳过"
  else
    cmd_eval
    ok "报告：$(ls -t reports/*.json 2>/dev/null | sed -n '1p' || echo '（见 reports/）')"
    ok "可读版：$(ls -t reports/*.md 2>/dev/null | sed -n '1p' || echo '（见 reports/）')"
  fi
  done_step

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
  echo "停止服务：bash scripts/local.sh stop"
  echo "查看状态：bash scripts/local.sh status"
  echo "评测报告：reports/"
  echo "运行日志：runs/agent_runs.jsonl（含 run_id 与每步 record_id/evidence）"
  echo
  echo "若只想本地看界面：bash scripts/local.sh ui  →  http://localhost:8501"
}

# ---------------------------------------------------------------- status / stop
cmd_status() {
  echo "项目目录：$PROJECT_DIR"
  echo "解释器：  $(locate_python)"
  echo "服务地址：$BASE_URL"
  if service_alive; then
    echo "服务状态：运行中（HTTP 200，PID $(service_pid || legacy_service_pid || echo '-')）"
    echo "API Key： $(api_key)"
  else
    echo "服务状态：未运行"
  fi
  echo
  print_document_records
}

cmd_stop() { stop_service; }

cmd_help() { sed -n '2,/^set -/p' "${BASH_SOURCE[0]}" | sed '$d'; }

# ---------------------------------------------------------------- 分发
case "${1:-check}" in
  check)  cmd_check ;;
  env)    cmd_env ;;
  data)   cmd_data ;;
  build)  cmd_build ;;
  api)    cmd_api ;;
  ui)     cmd_ui ;;
  demo)   cmd_demo ;;
  eval)   cmd_eval ;;
  verify) cmd_verify ;;
  run)    cmd_run ;;
  stop)   cmd_stop ;;
  status) cmd_status ;;
  -h|--help|help) cmd_help ;;
  *)
    echo "未知参数：$1（可用：check / env / data / build / api / ui / demo / eval / verify / run / stop / status / help）" >&2
    cmd_help >&2
    exit 1
    ;;
esac
