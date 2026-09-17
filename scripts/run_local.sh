#!/usr/bin/env bash
# run_local.sh —— 本地一键跑通知识库问答 Agent
#
# 只复用已有 Conda 环境，不创建环境、不安装依赖。
#
# 用法：
#   bash scripts/run_local.sh check     # 环境与数据自检（先跑这个）
#   bash scripts/run_local.sh env       # 生成 .env 与 API Key
#   bash scripts/run_local.sh data      # 准备 data/ 下的 PDF（可用 SAMPLE_PDF 指定样例）
#   bash scripts/run_local.sh build     # 建/重建向量知识库（首次会下载模型）
#   bash scripts/run_local.sh api       # 启动 FastAPI 服务（默认 127.0.0.1:8080）
#   bash scripts/run_local.sh ui        # 启动 Streamlit 界面（默认 8501）
#   bash scripts/run_local.sh demo      # check + env + data + build + 健康检查
#   bash scripts/run_local.sh eval      # 用固定问题集跑 Agent 评测
#   bash scripts/run_local.sh verify    # 语法检查 + 全量测试 + 离线冒烟
#
# 可用环境变量：CONDA_ENV_NAME / PYTHON_BIN / APP_PORT / SAMPLE_PDF
set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-ai_project}"
APP_PORT="${APP_PORT:-8080}"
# 可选：设置 SAMPLE_PDF 指向一个本地 PDF，data/ 为空时会自动复制进去。
SAMPLE_PDF="${SAMPLE_PDF:-}"

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
if [[ -z "${PYTHON_BIN:-}" || ! -x "$PYTHON_BIN" ]]; then
  echo "WARN: 未定位到 Conda 环境 $CONDA_ENV_NAME，回退到 python。" >&2
  PYTHON_BIN="python"
fi

banner() { echo; echo "======== $* ========"; }

# ---------------------------------------------------------------- env
cmd_env() {
  banner "准备 .env 与 API Key"
  if [[ ! -f .env ]]; then
    cp .env.example .env
    echo "   - 已从 .env.example 生成 .env"
  else
    echo "   - .env 已存在，保持不变"
  fi
  local key
  key="$(sed -n 's/^API_KEY=//p' .env | tail -n 1 | tr -d '\r')"
  if [[ -z "$key" ]]; then
    key="$("$PYTHON_BIN" -c 'import secrets;print(secrets.token_hex(32))')"
    if grep -q '^API_KEY=' .env; then
      sed -i "s|^API_KEY=.*|API_KEY=${key}|" .env
    else
      printf '\nAPI_KEY=%s\n' "$key" >> .env
    fi
    echo "   - 已生成随机 API_KEY 并写回 .env"
  else
    echo "   - .env 中已有 API_KEY"
  fi
}

# ---------------------------------------------------------------- data
cmd_data() {
  banner "准备知识库源文件 data/"
  mkdir -p data
  local count
  count="$( { find data -maxdepth 1 -iname '*.pdf' || true; } | wc -l | tr -d ' ')"
  if [[ "$count" -gt 0 ]]; then
    echo "   - data/ 已有 $count 个 PDF，直接使用"
    return 0
  fi
  if [[ -f "$SAMPLE_PDF" ]]; then
    cp "$SAMPLE_PDF" data/
    echo "   - data/ 为空，已从样例复制：$(basename "$SAMPLE_PDF")"
  else
    echo "   - data/ 为空，且未找到样例 PDF。" >&2
    echo "     请把自己的 PDF 放进 $PROJECT_DIR/data/ 后重新执行。" >&2
    return 1
  fi
}

# ---------------------------------------------------------------- check
cmd_check() {
  # data/ 可能还没建；先建出来，避免 find 在 pipefail 下中断脚本。
  mkdir -p data
  banner "1. 解释器与依赖"
  echo "   - Python: $("$PYTHON_BIN" -c 'import sys;print(sys.version.split()[0])') @ $PYTHON_BIN"
  "$PYTHON_BIN" - <<'PY'
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

  banner "2. 配置与数据"
  if [[ -f .env ]]; then
    echo "   - .env: 存在"
  else
    echo "   - .env: 缺失（执行 bash scripts/run_local.sh env 生成）"
  fi
  local pdfs
  pdfs="$( { find data -maxdepth 1 -iname '*.pdf' 2>/dev/null || true; } | wc -l | tr -d ' ')"
  echo "   - data/ 下 PDF: ${pdfs:-0} 个"

  banner "3. 知识库状态"
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path

from agentic_rag import Settings
from agentic_rag.knowledge_base import KnowledgeBase
from agentic_rag.models import ModelRuntime

settings = Settings.from_env()
kb = KnowledgeBase(settings, ModelRuntime(settings))
records = kb.document_records()
if not records:
    print("   - 目录为空：需要执行 bash scripts/run_local.sh build 建库")
for record in records:
    print(
        f"   - {record.source}: {record.status} / {record.chunk_count} 块 / "
        f"version={(record.active_version_id or '-')[:12]}"
    )
PY

  banner "4. 模型缓存"
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path

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

  banner "下一步"
  echo "   建库：bash scripts/run_local.sh build"
  echo "   起服务：bash scripts/run_local.sh api      （OpenAPI: http://127.0.0.1:${APP_PORT}/docs）"
  echo "   起界面：bash scripts/run_local.sh ui       （http://localhost:8501）"
  echo "   一条龙：bash scripts/run_local.sh demo"
}

# ---------------------------------------------------------------- build
cmd_build() {
  cmd_env
  cmd_data
  banner "构建向量知识库（--replace 重建）"
  "$PYTHON_BIN" create_db.py data --replace
}

# ---------------------------------------------------------------- api
cmd_api() {
  cmd_env
  local key
  key="$(sed -n 's/^API_KEY=//p' .env | tail -n 1 | tr -d '\r')"
  banner "启动 FastAPI：http://127.0.0.1:${APP_PORT}"
  echo "   - OpenAPI: http://127.0.0.1:${APP_PORT}/docs"
  echo "   - 健康检查: curl -H \"X-API-Key: ${key:0:8}...\" http://127.0.0.1:${APP_PORT}/v1/health/ready"
  echo "   - MCP: http://127.0.0.1:${APP_PORT}/mcp/"
  exec "$PYTHON_BIN" -m uvicorn api:app --host 127.0.0.1 --port "$APP_PORT"
}

# ---------------------------------------------------------------- ui
cmd_ui() {
  cmd_env
  banner "启动 Streamlit：http://localhost:8501"
  exec "$PYTHON_BIN" -m streamlit run app.py
}

# ---------------------------------------------------------------- eval
cmd_eval() {
  cmd_env
  banner "固定问题集评测（Agent 模式）"
  "$PYTHON_BIN" evaluate.py --mode agent --repeats 1
}

# ---------------------------------------------------------------- demo
cmd_demo() {
  cmd_check
  cmd_build
  banner "构建后自检"
  "$PYTHON_BIN" scripts/offline_smoke.py
  banner "完成"
  echo "   接着执行：bash scripts/run_local.sh api  或  bash scripts/run_local.sh ui"
}

# ---------------------------------------------------------------- verify
cmd_verify() {
  bash scripts/verify.sh
}

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
  *)
    sed -n '2,20p' "${BASH_SOURCE[0]}"
    exit 1
    ;;
esac
