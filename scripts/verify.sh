#!/usr/bin/env bash
# verify.sh —— 改造后的验证入口
#
# 只复用已有 Conda 环境，不创建环境、不安装依赖。
# 用法：
#   bash scripts/verify.sh
#   CONDA_ENV_NAME=ai_project bash scripts/verify.sh
#   PYTHON_BIN=/path/to/python bash scripts/verify.sh
set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-ai_project}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  # 优先从 conda env list 解析环境前缀（比 conda run 更稳，也不依赖环境已激活）。
  ENV_PREFIX=""
  if command -v conda >/dev/null 2>&1; then
    ENV_PREFIX="$(conda env list 2>/dev/null \
      | sed -n "s/^${CONDA_ENV_NAME}[[:space:]][[:space:]]*//p" \
      | head -n 1 | tr -d '\r')"
    ENV_PREFIX="${ENV_PREFIX//\\//}"
  fi
  for candidate in "$ENV_PREFIX/python.exe" "$ENV_PREFIX/bin/python"; do
    if [[ -n "$ENV_PREFIX" && -x "$candidate" ]]; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi
if [[ -z "${PYTHON_BIN:-}" || ! -x "$PYTHON_BIN" ]]; then
  echo "WARN: 未能定位 Conda 环境 $CONDA_ENV_NAME，回退到 python（需已激活对应环境）。" >&2
  PYTHON_BIN="python"
fi

echo "== 项目目录：$PROJECT_DIR"
echo "== 解释器：$PYTHON_BIN ($("$PYTHON_BIN" -c 'import sys;print(sys.version.split()[0])'))"

echo
echo "== [1/3] 语法检查"
"$PYTHON_BIN" -m compileall -q agentic_rag api.py tests scripts/offline_smoke.py
echo "   compileall 通过"

echo
echo "== [2/3] 自动化测试"
"$PYTHON_BIN" -m pytest -q -p no:cacheprovider

echo
echo "== [3/3] 离线端到端冒烟（不下载模型）"
"$PYTHON_BIN" scripts/offline_smoke.py

echo
echo "全部验证通过。"
