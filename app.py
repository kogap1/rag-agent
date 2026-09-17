from __future__ import annotations

import shutil
from pathlib import Path

import streamlit as st

from agentic_rag import AgenticRAG, Settings
from agentic_rag.knowledge_base import KnowledgeBase
from agentic_rag.models import ModelRuntime


st.set_page_config(page_title="Agentic RAG", page_icon="🧭", layout="wide")


@st.cache_resource
def build_services():
    settings = Settings.from_env()
    settings.ensure_directories()
    runtime = ModelRuntime(settings)
    kb = KnowledgeBase(settings, runtime)
    return settings, kb, AgenticRAG(runtime, kb, settings)


settings, kb, agent = build_services()

with st.sidebar:
    st.title("🧭 Agentic RAG")
    st.caption("规划 · 检索 · 重排 · 反思 · 引用")
    st.divider()
    st.subheader("知识库")
    try:
        documents = kb.list_documents()
        st.metric("已收录文档", len(documents))
        for name in documents:
            st.write(f"📄 {name}")
    except Exception as exc:
        st.warning(f"知识库尚未就绪：{exc}")

    uploaded = st.file_uploader("上传 PDF", type=["pdf"], accept_multiple_files=True)
    replace = st.checkbox("上传前清空原知识库", value=False)
    if st.button("构建知识库", type="primary", use_container_width=True, disabled=not uploaded):
        progress = st.progress(0, text="准备处理文档")
        total_chunks = 0
        try:
            if replace:
                kb.clear()
            for index, item in enumerate(uploaded or []):
                safe_name = Path(item.name).name
                target = settings.upload_dir / safe_name
                with target.open("wb") as file:
                    shutil.copyfileobj(item, file)
                total_chunks += kb.ingest(target, replace=False)
                progress.progress((index + 1) / len(uploaded), text=f"已处理 {safe_name}")
            st.success(f"完成：新增 {total_chunks} 个文本块")
            st.rerun()
        except Exception as exc:
            st.error(f"构建失败：{exc}")

    col1, col2 = st.columns(2)
    if col1.button("清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if col2.button("清空知识库", use_container_width=True):
        kb.clear()
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(f"设备：{agent.runtime.device.upper()}")
    st.caption(f"LLM：{settings.llm_model}")

st.title("本地知识库 Agent")
st.caption("回答过程可追踪，结论可回溯到文档页码。")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("trace"):
            with st.expander("查看 Agent 执行轨迹"):
                for step in message["trace"]:
                    icon = "✅" if step.get("status") == "completed" else "⚠️"
                    st.write(f"{icon} **{step['name']}** — {step['detail']}")
        if message.get("metrics"):
            metrics = message["metrics"]
            st.caption(
                f"延迟 {metrics['latency_ms'] / 1000:.2f}s · "
                f"工具调用 {metrics['tool_calls']} 次 · "
                f"上下文 {metrics['context_chars']} 字符"
            )
        if message.get("evidence"):
            with st.expander("查看引用证据"):
                for item in message["evidence"]:
                    st.markdown(f"**{item['citation']}** · 相关度 `{item['score']:.4f}`")
                    st.caption(item["content"])

if prompt := st.chat_input("询问知识库，例如：研究生综合测评的加分规则是什么？"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Agent 正在规划并调用工具…"):
            try:
                history = [
                    {"role": item["role"], "content": item["content"]}
                    for item in st.session_state.messages[:-1]
                ]
                result = agent.run(prompt, history)
                st.markdown(result.answer)
                with st.expander("查看 Agent 执行轨迹"):
                    for step in result.steps:
                        icon = "✅" if step.status == "completed" else "⚠️"
                        st.write(f"{icon} **{step.name}** — {step.detail}")
                st.caption(
                    f"延迟 {result.latency_ms / 1000:.2f}s · 工具调用 {len(result.tool_calls)} 次 · "
                    f"上下文 {result.context_chars} 字符 · 引用校验 {'通过' if result.grounded else '未通过'}"
                )
                if result.hits:
                    with st.expander("查看引用证据"):
                        for hit in result.hits:
                            st.markdown(f"**{hit.citation}** · 相关度 `{hit.score:.4f}`")
                            st.caption(hit.content)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result.answer,
                        "trace": [step.__dict__ for step in result.steps],
                        "evidence": [
                            {
                                "citation": hit.citation,
                                "score": hit.score,
                                "content": hit.content,
                            }
                            for hit in result.hits
                        ],
                        "metrics": {
                            "latency_ms": result.latency_ms,
                            "tool_calls": len(result.tool_calls),
                            "context_chars": result.context_chars,
                        },
                    }
                )
            except Exception as exc:
                st.error(f"Agent 执行失败：{exc}")
