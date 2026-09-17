from __future__ import annotations

import argparse
import json
from pathlib import Path

RATE_KEYS = {
    "task_success_rate",
    "grounding_rate",
    "tool_success_rate",
    "recovery_trigger_rate",
    "recovery_success_rate",
    "policy_violation_rate",
    "stability_at_n",
    "retrieval_recall_at_k",
    "context_compression_ratio",
}


def _fmt(key: str, value: float) -> str:
    if value is None:
        return "N/A"
    if key in RATE_KEYS:
        return f"{value:.2%}"
    if key in {"mean_cost_usd", "cost_per_success_usd"}:
        return f"${value:.6f}"
    return f"{value:.2f}"


def compare(first: Path, second: Path) -> str:
    a = json.loads(first.read_text(encoding="utf-8"))
    b = json.loads(second.read_text(encoding="utf-8"))
    keys = list(dict.fromkeys([*a["summary"].keys(), *b["summary"].keys()]))

    lines = [
        "# 评测报告对比",
        "",
        "| 指标 | {} | {} | 差值 |".format(first.name, second.name),
        "|---|---|---:|---:|",
    ]
    for key in keys:
        if key in {"runs", "cases", "repeats"}:
            continue
        av = a["summary"].get(key)
        bv = b["summary"].get(key)
        if not isinstance(av, (int, float)) or not isinstance(bv, (int, float)):
            continue
        delta = bv - av if isinstance(av, (int, float)) and isinstance(bv, (int, float)) else None
        delta_text = _fmt(key, delta) if delta is not None else "N/A"
        lines.append(f"| {key} | {_fmt(key, av)} | {_fmt(key, bv)} | {delta_text} |")

    meta = ["", "## 实验条件（两个报告可能不同，务必核对）", ""]
    for key in {"mode", "tag", "model", "embedding", "reranker", "device", "repeats", "dataset"}:
        av = a["metadata"].get(key)
        bv = b["metadata"].get(key)
        marker = "（不同！）" if av != bv else ""
        meta.append(f"- {key}: `{av}` → `{bv}` {marker}".rstrip())
    return "\n".join(lines + meta) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="对比两份评测报告（如 baseline vs agent、reranker on vs off）")
    parser.add_argument("report_a", type=Path)
    parser.add_argument("report_b", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="写出 Markdown 对比文件")
    args = parser.parse_args()
    text = compare(args.report_a, args.report_b)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"对比已写入：{args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
