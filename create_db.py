from __future__ import annotations

import argparse
from pathlib import Path

from agentic_rag import Settings
from agentic_rag.knowledge_base import KnowledgeBase
from agentic_rag.models import ModelRuntime


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 Agentic RAG 向量知识库")
    parser.add_argument("paths", nargs="+", type=Path, help="一个或多个 PDF 文件/目录")
    parser.add_argument("--replace", action="store_true", help="构建前清空现有知识库")
    args = parser.parse_args()

    settings = Settings.from_env()
    settings.ensure_directories()
    kb = KnowledgeBase(settings, ModelRuntime(settings))
    if args.replace:
        kb.clear()

    files: list[Path] = []
    for path in args.paths:
        files.extend(sorted(path.glob("*.pdf")) if path.is_dir() else [path])
    if not files:
        raise SystemExit("没有找到 PDF 文件")

    total = 0
    for file in files:
        count = kb.ingest(file)
        total += count
        print(f"{file.name}: 新增 {count} 个文本块")
    print(f"完成，共新增 {total} 个文本块。")


if __name__ == "__main__":
    main()
