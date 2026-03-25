from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

import streamlit as st

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ingest import DATA_DIR
from rag_pipeline import ask_question
from retriever import build_vector_store


st.set_page_config(
    page_title="Doc RAG Assistant",
    layout="wide",
)


HISTORY_CONTAINER_CSS = """
<style>
    .history-shell {
        max-height: 72vh;
        overflow-y: auto;
        padding-right: 0.5rem;
    }
    .history-card {
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
        background: rgba(248, 250, 252, 0.72);
    }
    .history-card h4 {
        margin: 0 0 0.35rem 0;
    }
    .history-meta {
        font-size: 0.92rem;
        color: #475569;
        margin-bottom: 0.65rem;
    }
</style>
"""


def ensure_session_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "saved_upload_tokens" not in st.session_state:
        st.session_state.saved_upload_tokens = set()


def save_uploaded_pdf(uploaded_file) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target_path = DATA_DIR / uploaded_file.name
    target_path.write_bytes(uploaded_file.getbuffer())
    return target_path


def format_sources(sources: dict[int, tuple[str, int | str]]) -> str:
    if not sources:
        return "(none)"

    lines = []
    for citation_id, (source, page) in sources.items():
        lines.append(f"- [{citation_id}] {source} | page {page}")
    return "\n".join(lines)


def add_history_entry(question: str, result: dict) -> None:
    st.session_state.chat_history.append(
        {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "question": question,
            "answer": result.get("answer", ""),
            "query_type": result.get("query_type", "UNKNOWN"),
            "routing": result.get("routing", {}),
            "sources": result.get("sources", {}),
        }
    )


def render_history() -> None:
    st.markdown(HISTORY_CONTAINER_CSS, unsafe_allow_html=True)
    st.markdown('<div class="history-shell">', unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.info("Ask a question to start the answer history.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    for entry in reversed(st.session_state.chat_history):
        routing = entry.get("routing", {})
        sources = entry.get("sources", {})

        st.markdown('<div class="history-card">', unsafe_allow_html=True)
        st.markdown(f"#### {entry['question']}")
        st.markdown(
            (
                '<div class="history-meta">'
                f"{entry['timestamp']} | Route: {entry['query_type']} | "
                f"Source: {routing.get('route_source', 'UNKNOWN')} | "
                f"Confidence: {routing.get('confidence', 'n/a')}"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

        reason = routing.get("reason")
        if reason:
            st.caption(f"Reason: {reason}")

        st.markdown("**Answer**")
        st.write(entry["answer"])

        st.markdown("**Citations & Sources**")
        if sources:
            st.markdown(format_sources(sources))
        elif entry["query_type"] == "CONVERSATION":
            st.caption("Not applicable for conversation mode.")
        else:
            st.caption("(none)")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    ensure_session_state()

    st.title("Document RAG Assistant")
    st.caption("Upload PDF documents, rebuild the index, and query them with route-aware answers.")

    left_col, right_col = st.columns([1, 1.55], gap="large")

    with left_col:
        st.subheader("Documents")
        uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
        if uploaded_file is not None:
            upload_token = (uploaded_file.name, uploaded_file.size)
            if upload_token not in st.session_state.saved_upload_tokens:
                saved_path = save_uploaded_pdf(uploaded_file)
                st.session_state.saved_upload_tokens.add(upload_token)
                st.success(f"Saved `{saved_path.name}` to `{DATA_DIR}`.")

        pdf_files = sorted(DATA_DIR.glob("*.pdf"))
        if pdf_files:
            st.markdown("**Available PDFs**")
            for pdf_file in pdf_files:
                st.write(f"- {pdf_file.name}")
        else:
            st.info("No PDFs found yet. Upload a file to get started.")

        if st.button("Rebuild Index", use_container_width=True):
            with st.spinner("Indexing documents..."):
                try:
                    build_vector_store()
                except Exception as exc:
                    st.error(f"Failed to rebuild index: {exc}")
                else:
                    st.success("Vector store rebuilt successfully.")

        st.divider()
        st.subheader("Ask a Question")
        question = st.text_area(
            "Query",
            placeholder="Ask about the uploaded documents...",
            height=140,
        )

        if st.button("Submit Query", type="primary", use_container_width=True):
            if not question.strip():
                st.warning("Enter a question first.")
            else:
                with st.spinner("Generating answer..."):
                    try:
                        result = ask_question(question.strip())
                    except Exception as exc:
                        st.error(f"Query failed: {exc}")
                    else:
                        add_history_entry(question.strip(), result)
                        st.success("Answer added to history.")

        if st.button("Clear History", use_container_width=True):
            st.session_state.chat_history = []
            st.success("Cleared answer history.")

    with right_col:
        history_tab, latest_tab = st.tabs(["Answer History", "Latest Response"])

        with history_tab:
            st.subheader("Scrollable Answer History")
            render_history()

        with latest_tab:
            st.subheader("Most Recent Response")
            if st.session_state.chat_history:
                latest = st.session_state.chat_history[-1]
                st.markdown(f"### {latest['question']}")
                st.write(latest["answer"])

                routing = latest.get("routing", {})
                st.markdown("**Route Metadata**")
                st.json(
                    {
                        "query_type": latest.get("query_type", "UNKNOWN"),
                        "route_source": routing.get("route_source", "UNKNOWN"),
                        "confidence": routing.get("confidence", "n/a"),
                        "reason": routing.get("reason", ""),
                    }
                )

                st.markdown("**Citations & Sources**")
                sources = latest.get("sources", {})
                if sources:
                    st.markdown(format_sources(sources))
                elif latest.get("query_type") == "CONVERSATION":
                    st.caption("Not applicable for conversation mode.")
                else:
                    st.caption("(none)")
            else:
                st.info("No response yet. Submit a query to populate this panel.")


if __name__ == "__main__":
    main()
