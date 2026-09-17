"""Agentic RAG core package with lazy public imports."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import AgenticRAG
    from .config import Settings

__all__ = ["AgenticRAG", "Settings"]


def __getattr__(name: str):
    if name == "AgenticRAG":
        from .agent import AgenticRAG

        return AgenticRAG
    if name == "Settings":
        from .config import Settings

        return Settings
    raise AttributeError(name)
