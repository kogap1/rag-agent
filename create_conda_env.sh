#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# ---------------------------------------------------------------- 失败可见性
# set -e 下失败命令直接退出，且很多命令自身不打印可读信息，表现为「跑一半静默回到
# 提示符」。装上 ERR trap，把行号、失败命令、退出码打出来。
trap 'err_status=$?; err_line=$LINENO; err_cmd=$BASH_COMMAND; if [[ "$err_line" != "${ERR_SEEN_LINE:-}" ]]; then ERR_SEEN_LINE="$err_line"; printf "\n   ❌ 环境准备中断：第 %s 行执行失败（退出码 %s）\n" "$err_line" "$err_status" >&2; printf "      失败命令：%s\n" "$err_cmd" >&2; printf "      文件位置：%s\n" "${BASH_SOURCE[0]}" >&2; printf "      查看该行：sed -n %sp %s\n" "$err_line" "${BASH_SOURCE[0]}" >&2; fi; true' ERR

CONDA_ENV_NAME="${CONDA_ENV_NAME:-rag-agent-cu11x}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu118}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
INSTALL_DEV="${INSTALL_DEV:-0}"
MIN_LINUX_DRIVER="450.80.02"
CONDA_CHANNEL="${CONDA_CHANNEL:-defaults}"

command -v conda >/dev/null 2>&1 || {
  echo "ERROR: 未找到 conda，请先安装 Miniconda/Anaconda。" >&2
  exit 1
}

echo "conda 版本: $(conda --version 2>&1)"

version_ge() {
  printf '%s\n%s\n' "$2" "$1" | sort -V -C
}

# 解析环境前缀下的解释器。
# 刻意不使用 `conda run`：它的 --no-capture-output 需要较新的 conda（老版本会报
# "unrecognized arguments: --no-capture-output"），且 conda run 会捕获/缓冲子进程
# 输出，导致安装过程看不到实时进度。直接调用环境自带的解释器最稳，也不依赖环境已激活。
# sed -n '1p' 而非 head -n 1：head 提前退出会让上游收到 SIGPIPE，在 pipefail 下
# 整条管道被判失败。
resolve_env_python() {
  local env_name="$1" prefix candidate
  command -v conda >/dev/null 2>&1 || return 1
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

if [[ "$REQUIRE_CUDA" == "1" ]]; then
  command -v nvidia-smi >/dev/null 2>&1 || {
    echo "ERROR: REQUIRE_CUDA=1，但未找到 nvidia-smi。" >&2
    exit 1
  }
  DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | sed -n '1p' | tr -d '[:space:]' || true)"
  if [[ -z "$DRIVER_VERSION" ]]; then
    echo "ERROR: 无法从 nvidia-smi 读取驱动版本，请手工确认 GPU 状态。" >&2
    exit 1
  fi
  if ! version_ge "$DRIVER_VERSION" "$MIN_LINUX_DRIVER"; then
    echo "ERROR: NVIDIA 驱动 $DRIVER_VERSION 低于 CUDA 11.x minor compatibility 要求 $MIN_LINUX_DRIVER。" >&2
    exit 1
  fi
  echo "检测到 NVIDIA 驱动: $DRIVER_VERSION"
fi

# 环境是否已存在。用 awk 收集结果并统一退出：awk 会读完整个输入，
# 不会像 `grep -q` 那样提前退出导致上游收到 SIGPIPE。
if conda env list 2>/dev/null | awk -v name="$CONDA_ENV_NAME" '$1 == name { found = 1 } END { exit(found ? 0 : 1) }'; then
  echo "复用 Conda 环境: $CONDA_ENV_NAME"
else
  create_args=(
    create -y -n "$CONDA_ENV_NAME" "python=$PYTHON_VERSION" pip
    --override-channels -c "$CONDA_CHANNEL"
  )
  if command -v mamba >/dev/null 2>&1; then
    echo "使用 mamba 创建 Conda 环境。"
    mamba "${create_args[@]}"
  elif conda create --help 2>&1 | grep -q -- '--solver'; then
    echo "优先使用 libmamba 求解器创建 Conda 环境。"
    if ! conda "${create_args[@]}" --solver=libmamba; then
      echo "WARN: libmamba 不可用，回退到 classic 求解器。" >&2
      conda "${create_args[@]}" --solver=classic
    fi
  else
    echo "当前 Conda 不支持 --solver，使用 classic 求解器。"
    conda "${create_args[@]}"
  fi
fi

ENV_PYTHON="$(resolve_env_python "$CONDA_ENV_NAME" || true)"
if [[ -z "$ENV_PYTHON" || ! -x "$ENV_PYTHON" ]]; then
  echo "ERROR: 无法定位 Conda 环境 $CONDA_ENV_NAME 的 Python 解释器。" >&2
  echo "请检查: conda env list" >&2
  exit 1
fi
echo "环境解释器: $ENV_PYTHON"

ENV_VERSION="$("$ENV_PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$ENV_VERSION" != "$PYTHON_VERSION" ]]; then
  echo "ERROR: Conda 环境 $CONDA_ENV_NAME 的 Python 为 $ENV_VERSION，需要 $PYTHON_VERSION。" >&2
  echo "请换一个 CONDA_ENV_NAME，或手工删除旧环境后重试。" >&2
  exit 1
fi

if ! "$ENV_PYTHON" -m pip --version >/dev/null 2>&1; then
  echo "检测到环境缺少 pip，正在自动修复。"
  if ! "$ENV_PYTHON" -m ensurepip --upgrade; then
    echo "ensurepip 修复失败，改为强制重装 Conda pip 包。" >&2
    conda install -y -n "$CONDA_ENV_NAME" --force-reinstall \
      pip setuptools wheel --override-channels -c "$CONDA_CHANNEL"
  fi
fi

"$ENV_PYTHON" -m pip install --upgrade pip setuptools wheel

# CUDA 11.1 对应的最后一个官方 PyTorch wheel 是 1.10.1，无法满足当前
# LangGraph/MCP/Transformers 栈。这里安装 PyTorch 2.6 官方 cu118 wheel；
# 它自带 CUDA runtime，并在 CUDA 11.x 驱动上使用 NVIDIA minor compatibility。
"$ENV_PYTHON" -m pip install \
  "torch==$TORCH_VERSION" \
  --index-url "$TORCH_INDEX_URL"

"$ENV_PYTHON" -m pip install -r requirements.txt
if [[ "$INSTALL_DEV" == "1" ]]; then
  "$ENV_PYTHON" -m pip install -r requirements-dev.txt
fi
"$ENV_PYTHON" -m pip check

runtime_check="$(cat <<'PY'
import os
import torch
import fastapi
import langgraph
import mcp
import sentence_transformers
import transformers

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
)"
REQUIRE_CUDA="$REQUIRE_CUDA" "$ENV_PYTHON" -c "$runtime_check"

echo "Conda 环境创建/校验完成: $CONDA_ENV_NAME"
echo "解释器: $ENV_PYTHON"
