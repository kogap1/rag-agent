from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

COMBOS = [
    ("baseline", False),
    ("baseline", True),
    ("agent", False),
    ("agent", True),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="一键运行 baseline/agent × reranker on/off 消融矩阵，产出四份评测报告"
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dataset", type=Path, default=Path("eval/dataset.example.jsonl"))
    args = parser.parse_args()

    reports: list[Path] = []
    for mode, disable_reranker in COMBOS:
        command = [
            sys.executable,
            "evaluate.py",
            "--mode",
            mode,
            "--dataset",
            str(args.dataset),
            "--repeats",
            str(args.repeats),
        ]
        if disable_reranker:
            command.append("--disable-reranker")
        print(f">>> {' '.join(command)}")
        subprocess.run(command, check=True)
        # 找到刚生成的对应报告（取最新）
        matches = sorted(Path("reports").glob(f"{mode}-*.json"), key=lambda p: p.stat().st_mtime)
        if matches:
            reports.append(matches[-1])

    print("\n# 消融矩阵完成，报告：")
    for report in reports:
        print(f"- {report}")
    print("\n# 对比示例：")
    if len(reports) == 4:
        print(f"python compare_reports.py {reports[0]} {reports[2]}  # baseline vs agent（reranker on）")
        print(f"python compare_reports.py {reports[0]} {reports[1]}  # reranker on vs off（baseline）")
        print(f"python compare_reports.py {reports[2]} {reports[3]}  # reranker on vs off（agent）")


if __name__ == "__main__":
    main()
