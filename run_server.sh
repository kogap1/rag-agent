#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# 保存命令行传入值；.env 只提供默认值，不能覆盖显式传参。
DEPLOY_MODE_OVERRIDE="${DEPLOY_MODE-}"
CONDA_ENV_NAME_OVERRIDE="${CONDA_ENV_NAME-}"
REQUIRE_CUDA_OVERRIDE="${REQUIRE_CUDA-}"
PUBLIC_HOST_OVERRIDE="${PUBLIC_HOST-}"

APP_PORT="${APP_PORT:-8080}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-180}"
PUBLIC_HOST="${PUBLIC_HOST:-}"
DEPLOY_MODE="${DEPLOY_MODE:-conda}"

command -v curl >/dev/null 2>&1 || {
  echo "ERROR: curl 未安装，无法执行启动验收" >&2
  exit 1
}

if [[ ! -f .env ]]; then
  cp .env.example .env
fi

current_api_key="$(sed -n 's/^API_KEY=//p' .env | tail -n 1 | tr -d '\r')"
if [[ -z "$current_api_key" ]]; then
  if command -v openssl >/dev/null 2>&1; then
    generated_api_key="$(openssl rand -hex 32)"
  elif command -v python3 >/dev/null 2>&1; then
    generated_api_key="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
  else
    echo "ERROR: 无法生成 API Key，需要 openssl 或 python3。" >&2
    exit 1
  fi
  if grep -q '^API_KEY=' .env; then
    sed -i "s/^API_KEY=.*/API_KEY=${generated_api_key}/" .env
  else
    printf '\nAPI_KEY=%s\n' "$generated_api_key" >> .env
  fi
  chmod 600 .env
  echo "已自动生成 API Key 并写入 .env（文件权限 600）。"
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

DEPLOY_MODE="${DEPLOY_MODE_OVERRIDE:-${DEPLOY_MODE:-conda}}"
PUBLIC_HOST="${PUBLIC_HOST_OVERRIDE:-${PUBLIC_HOST:-}}"
REQUIRE_CUDA="${REQUIRE_CUDA_OVERRIDE:-${REQUIRE_CUDA:-1}}"

# 显式 CONDA_ENV_NAME 优先；否则优先复用当前已激活的非 base 环境，
# 最后才读取 .env。启动脚本绝不创建环境或安装/升级依赖。
if [[ -n "$CONDA_ENV_NAME_OVERRIDE" ]]; then
  CONDA_ENV_NAME="$CONDA_ENV_NAME_OVERRIDE"
elif [[ -n "${CONDA_DEFAULT_ENV:-}" && "${CONDA_DEFAULT_ENV}" != "base" ]]; then
  CONDA_ENV_NAME="$CONDA_DEFAULT_ENV"
else
  CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"
fi

API_KEY="${API_KEY:-}"
if [[ -z "$API_KEY" ]]; then
  echo "ERROR: .env 中 API_KEY 不能为空；服务器部署必须使用强随机值。" >&2
  echo "可生成：openssl rand -hex 32" >&2
  exit 1
fi

MCP_ALLOWED_HOSTS="${MCP_ALLOWED_HOSTS:-127.0.0.1:*,localhost:*,[::1]:*}"
if [[ -n "$PUBLIC_HOST" && ",${MCP_ALLOWED_HOSTS}," != *",${PUBLIC_HOST},"* ]]; then
  MCP_ALLOWED_HOSTS="${MCP_ALLOWED_HOSTS},${PUBLIC_HOST},${PUBLIC_HOST}:*"
fi
MCP_ALLOWED_ORIGINS="${MCP_ALLOWED_ORIGINS:-http://127.0.0.1:*,http://localhost:*,http://[::1]:*}"

export APP_PORT API_KEY MCP_ALLOWED_HOSTS MCP_ALLOWED_ORIGINS

mkdir -p uploaded_files server_vector_db state runs models

case "$DEPLOY_MODE" in
  conda)
    command -v conda >/dev/null 2>&1 || {
      echo "ERROR: 未找到 conda。" >&2
      exit 1
    }

    if [[ -z "$CONDA_ENV_NAME" || "$CONDA_ENV_NAME" == "base" ]]; then
      echo "ERROR: 未指定可复用的 Conda 环境。" >&2
      echo "请先执行 conda activate <已有环境>，再运行 ./run_server.sh；" >&2
      echo "或执行 CONDA_ENV_NAME=<已有环境> ./run_server.sh。脚本不会自动下载依赖。" >&2
      exit 1
    fi

    # 用 conda env list 解析环境前缀下的解释器，不依赖 `conda run`：
    # 老版本 conda 不支持 conda run 的 --no-capture-output，且 conda run 会缓冲
    # 子进程输出；直接调用环境自带的解释器最稳，也不要求环境已激活。
    env_prefix="$(conda env list 2>/dev/null \
      | awk -v name="$CONDA_ENV_NAME" '$1 == name { print $NF }' \
      | sed -n '1p' | tr -d '\r' || true)"
    env_prefix="${env_prefix//\\//}"
    env_python=""
    for candidate in "$env_prefix/bin/python" "$env_prefix/python.exe"; do
      if [[ -n "$env_prefix" && -x "$candidate" ]]; then
        env_python="$candidate"
        break
      fi
    done
    if [[ -z "$env_python" || ! -x "$env_python" ]]; then
      echo "ERROR: Conda 环境不存在或无法解析解释器: $CONDA_ENV_NAME" >&2
      echo "可用环境请查看: conda env list" >&2
      exit 1
    fi

    echo "复用已有 Conda 环境: $CONDA_ENV_NAME"
    echo "Python: $env_python"
    echo "离线检查项目依赖（不会执行 pip/conda install）"
    if ! dependency_check="$({ REQUIRE_CUDA="$REQUIRE_CUDA" "$env_python" - <<'PY'
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
)
missing = [name for name in required_modules if importlib.util.find_spec(name) is None]
if missing:
    print("缺少模块: " + ", ".join(missing), file=sys.stderr)
    raise SystemExit(2)

import torch

if os.environ.get("REQUIRE_CUDA", "1") == "1" and not torch.cuda.is_available():
    print("PyTorch 已安装，但当前环境无法访问 CUDA。", file=sys.stderr)
    raise SystemExit(3)

import api

print(f"依赖检查通过: Python {sys.version.split()[0]}, torch {torch.__version__}")
PY
    } 2>&1)"; then
      echo "ERROR: 已有环境 $CONDA_ENV_NAME 不能直接运行本项目。" >&2
      echo "$dependency_check" >&2
      echo "启动已停止，未下载或修改任何包。请改用一个已经装好依赖的环境。" >&2
      exit 1
    fi
    echo "$dependency_check"

    if [[ -f runs/uvicorn.pid ]]; then
      existing_pid="$(cat runs/uvicorn.pid)"
      if kill -0 "$existing_pid" >/dev/null 2>&1; then
        echo "ERROR: 服务进程已存在，PID=$existing_pid。" >&2
        exit 1
      fi
    fi
    nohup "$env_python" -m uvicorn api:app --host 0.0.0.0 --port "$APP_PORT" --workers 1 \
      > runs/uvicorn.log 2>&1 &
    service_pid=$!
    printf '%s\n' "$service_pid" > runs/uvicorn.pid
    ;;
  docker)
    command -v docker >/dev/null 2>&1 || {
      echo "ERROR: docker 未安装或不在 PATH 中" >&2
      exit 1
    }
    docker compose version >/dev/null 2>&1 || {
      echo "ERROR: docker compose 插件不可用" >&2
      exit 1
    }
    docker compose config --quiet
    docker compose up --build -d --remove-orphans
    ;;
  *)
    echo "ERROR: DEPLOY_MODE 只能是 conda 或 docker。" >&2
    exit 1
    ;;
esac

deadline=$((SECONDS + READY_TIMEOUT_SECONDS))
live_url="http://127.0.0.1:${APP_PORT}/v1/health/live"
ready_url="http://127.0.0.1:${APP_PORT}/v1/health/ready"

until curl --silent --show-error --fail "$live_url" >/dev/null 2>&1; do
  if (( SECONDS >= deadline )); then
    echo "ERROR: 服务未在 ${READY_TIMEOUT_SECONDS}s 内通过存活检查" >&2
    if [[ "$DEPLOY_MODE" == "conda" ]]; then
      tail -n 80 runs/uvicorn.log >&2 || true
      kill "$service_pid" >/dev/null 2>&1 || true
    else
      docker compose logs --tail=80 agentic-rag >&2
    fi
    exit 1
  fi
  sleep 2
done

until curl --silent --show-error --fail \
  -H "X-API-Key: ${API_KEY}" "$ready_url" >/dev/null 2>&1; do
  if (( SECONDS >= deadline )); then
    echo "ERROR: 服务未在 ${READY_TIMEOUT_SECONDS}s 内通过就绪检查" >&2
    if [[ "$DEPLOY_MODE" == "conda" ]]; then
      tail -n 80 runs/uvicorn.log >&2 || true
      kill "$service_pid" >/dev/null 2>&1 || true
    else
      docker compose logs --tail=80 agentic-rag >&2
    fi
    exit 1
  fi
  sleep 2
done

echo "Agentic RAG 已启动并通过健康检查"
echo "REST API: http://<server>:${APP_PORT}/docs"
echo "MCP:      http://<server>:${APP_PORT}/mcp/"
echo "Metrics:  http://<server>:${APP_PORT}/metrics"
if [[ "$DEPLOY_MODE" == "conda" ]]; then
  echo "Conda环境: $CONDA_ENV_NAME"
  echo "进程PID:   $(cat runs/uvicorn.pid)"
  echo "查看日志: tail -f runs/uvicorn.log"
  echo "停止服务: kill $(cat runs/uvicorn.pid)"
else
  echo "查看日志: docker compose logs -f agentic-rag"
fi
