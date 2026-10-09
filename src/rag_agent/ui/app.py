"""
app.py
======
Streamlit user interface for the Deep Learning RAG Interview Prep Agent.

Three-panel layout:
  - Left sidebar: Document ingestion and corpus browser
  - Centre: Document viewer
  - Right: Chat interface

API contract with the backend (agree this with Pipeline Engineer
before building anything):

  ingest(file_paths: list[Path]) -> IngestionResult
  list_documents() -> list[dict]
  get_document_chunks(source: str) -> list[DocumentChunk]
  chat(query: str, history: list[dict], filters: dict) -> AgentResponse

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from rag_agent.agent.graph import get_compiled_graph
from rag_agent.agent.state import AgentResponse
from rag_agent.config import get_settings
from rag_agent.corpus.chunker import DocumentChunker
from rag_agent.vectorstore.store import VectorStoreManager


# ---------------------------------------------------------------------------
# Cached Resources
# ---------------------------------------------------------------------------
# Use st.cache_resource for objects that should persist across reruns
# and be shared across all user sessions. This prevents re-initialising
# ChromaDB and reloading the embedding model on every button click.


@st.cache_resource
def get_vector_store() -> VectorStoreManager:
    """
    Return the singleton VectorStoreManager.

    Cached so ChromaDB connection is initialised once per application
    session, not on every Streamlit rerun.
    """
    return VectorStoreManager()


@st.cache_resource
def get_chunker() -> DocumentChunker:
    """Return the singleton DocumentChunker."""
    return DocumentChunker()


@st.cache_resource
def get_graph():
    """Return the compiled LangGraph agent."""
    return get_compiled_graph()


@st.cache_resource
def get_llm():
    """Load and cache the Groq language model."""
    from rag_agent.config import LLMFactory

    return LLMFactory().create()


# ---------------------------------------------------------------------------
# Session State Initialisation
# ---------------------------------------------------------------------------


def initialise_session_state() -> None:
    """
    Initialise all st.session_state keys on first run.

    Must be called at the top of main() before any UI is rendered.
    Without this, state keys referenced in callbacks will raise KeyError.

    Interview talking point: Streamlit reruns the entire script on every
    user interaction. session_state is the mechanism for persisting data
    (chat history, ingestion results) across reruns.
    """
    defaults = {
        "chat_history": [],           # list of {"role": "user"|"assistant", "content": str}
        "ingested_documents": [],     # list of dicts from list_documents()
        "selected_document": None,    # source filename currently in viewer
        "last_ingestion_result": None,
        "thread_id": "default-session",  # LangGraph conversation thread
        "topic_filter": None,
        "difficulty_filter": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ---------------------------------------------------------------------------
# Ingestion Panel (Sidebar)
# ---------------------------------------------------------------------------


def render_ingestion_panel(
    store: VectorStoreManager,
    chunker: DocumentChunker,
) -> None:
    """
    Render the document ingestion panel in the sidebar.

    Allows multi-file upload of PDF and Markdown files. Displays
    ingestion results (chunks added, duplicates skipped, errors).
    Updates the ingested documents list after successful ingestion.

    Parameters
    ----------
    store : VectorStoreManager
    chunker : DocumentChunker
    """
    st.sidebar.header("📂 Corpus Ingestion")

    
    import tempfile

    st.sidebar.header("📂 Corpus Ingestion")

    uploaded_files = st.sidebar.file_uploader(
        "Upload Markdown documents",
        type=["md"],
        accept_multiple_files=True,
    )

    if st.sidebar.button(
        "Ingest Documents",
        disabled=not uploaded_files,
    ):
        all_chunks = []

        with tempfile.TemporaryDirectory() as temp_dir:
            for uploaded_file in uploaded_files:
                file_path = Path(temp_dir) / uploaded_file.name
                file_path.write_bytes(uploaded_file.getvalue())

                chunks = chunker.chunk_file(file_path)
                all_chunks.extend(chunks)

            result = store.ingest(all_chunks)

        st.sidebar.success(
            f"{result.ingested} chunks added, "
            f"{result.skipped} duplicates skipped."
        )

        if result.errors:
            st.sidebar.error(
                f"{result.errors} chunks failed."
            )

    st.sidebar.metric(
        "Total Chunks",
        store._collection.count(),
    )

    st.sidebar.info("Upload .pdf or .md files to populate the corpus.")


def render_corpus_stats(store: VectorStoreManager) -> None:
    """
    Render a compact corpus health summary in the sidebar.

    Shows total chunks, topics covered, and whether bonus topics
    are present. Used during Hour 3 to demonstrate corpus completeness.

    Parameters
    ----------
    store : VectorStoreManager
    """
    
    st.sidebar.write(
        "Stored document chunks:",
        store._collection.count(),
    )

    pass


# ---------------------------------------------------------------------------
# Document Viewer Panel (Centre)
# ---------------------------------------------------------------------------


def render_document_viewer(store: VectorStoreManager) -> None:
    """
    Render the document viewer in the main centre column.

    Displays a selectable list of ingested documents. When a document
    is selected, renders its chunk content in a scrollable pane.

    Parameters
    ----------
    store : VectorStoreManager
    """
    st.subheader("📄 Document Viewer")

    
    st.subheader("📄 Document Viewer")

    data = store._collection.get(
        include=["documents", "metadatas"]
    )

    if not data["ids"]:
        st.info("Upload a Markdown document to begin.")
        return

    sources = sorted({
        metadata["source"]
        for metadata in data["metadatas"]
    })

    selected_source = st.selectbox(
        "Select document",
        sources,
    )

    with st.container(height=400):
        for text, metadata in zip(
            data["documents"],
            data["metadatas"],
        ):
            if metadata["source"] == selected_source:
                st.markdown(text)
                st.divider()

    st.info("Ingest documents using the sidebar to view content here.")


# ---------------------------------------------------------------------------
# Chat Interface Panel (Right)
# ---------------------------------------------------------------------------

def render_chat_interface(graph) -> None:
    """Display the RAG chatbot."""
    from langchain_core.messages import HumanMessage

    st.subheader("💬 Interview Prep Chat")

    store = get_vector_store()

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                st.caption(
                    "Sources: " + ", ".join(message["sources"])
                )

    question = st.chat_input(
        "Ask a question about your documents..."
    )

    if question:
        st.session_state.chat_history.append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                chunks = store.query(question)

                if not chunks:
                    answer = (
                        "No relevant information was found "
                        "in the uploaded documents."
                    )
                    sources = []
                else:
                    context = "\n\n".join(
                        chunk.chunk_text for chunk in chunks
                    )

                    prompt = (
                        "Answer the question using only the "
                        "following document context. "
                        "If the answer is not present, say so.\n\n"
                        f"Context:\n{context}\n\n"
                        f"Question: {question}"
                    )

                    llm = get_llm()
                    response = llm.invoke([
                        HumanMessage(content=prompt)
                    ])

                    answer = response.content
                    sources = list({
                        chunk.metadata.source
                        for chunk in chunks
                    })

                st.markdown(answer)

                if sources:
                    st.caption(
                        "Sources: " + ", ".join(sources)
                    )

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Application entry point.

    Sets page config, initialises session state, instantiates shared
    resources, and renders all UI panels.

    Run with: uv run streamlit run src/rag_agent/ui/app.py
    """
    settings = get_settings()

    st.set_page_config(
        page_title=settings.app_title,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title(f"🧠 {settings.app_title}")
    st.caption(
        "RAG-powered interview preparation — built with LangChain, LangGraph, and ChromaDB"
    )

    initialise_session_state()

    # Instantiate shared backend resources
    store = get_vector_store()
    chunker = get_chunker()
    graph = None

    # Sidebar
    render_ingestion_panel(store, chunker)
    render_corpus_stats(store)

    # Main content area — two columns
    viewer_col, chat_col = st.columns([1, 1], gap="large")

    with viewer_col:
        render_document_viewer(store)

    with chat_col:
        render_chat_interface(graph)


if __name__ == "__main__":
    main()
