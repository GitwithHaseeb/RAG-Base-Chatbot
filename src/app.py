"""Streamlit app for GH Buddy RAG chatbot."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List

import streamlit as st
from dotenv import load_dotenv

from create_vectorstore import add_documents_to_vectorstore, reset_vectorstore
from load_documents import load_documents
from rag_chain import generate_answer
from utils import append_message, clear_messages, init_session_state, log_chat


load_dotenv()
st.set_page_config(page_title="GH Buddy", page_icon="🤖", layout="wide")
init_session_state()


def _render_logo() -> None:
    logo_path = Path("logo/gh_logo.png")
    if logo_path.exists():
        st.sidebar.image(str(logo_path), caption="GH Buddy")


def _save_uploaded_files(uploaded_files) -> List[str]:
    saved_paths: List[str] = []
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    for file in uploaded_files:
        suffix = Path(file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file.getvalue())
            temp_path = Path(tmp.name)
        final_path = data_dir / file.name
        final_path.write_bytes(temp_path.read_bytes())
        saved_paths.append(str(final_path))
    return saved_paths


st.title("GH Buddy – AI Assistant for Students")
st.caption("Ask questions using your own notes, syllabus, and URLs.")

_render_logo()

with st.sidebar:
    st.header("Ingestion")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, PPTX, or TXT files",
        type=["pdf", "docx", "pptx", "txt"],
        accept_multiple_files=True,
    )
    url_input = st.text_area("Add URLs (one per line)", placeholder="https://example.com/page")
    replace_existing = st.checkbox("Replace existing knowledge when processing", value=True)
    model_choice = st.toggle("Model: Gemini 🤖 / Hugging Face 🤗", value=False)
    st.session_state.model_choice = "Hugging Face" if model_choice else "Gemini"
    st.write(f"Selected model: **{st.session_state.model_choice}**")

    if st.button("Process & Add to Vector Store", use_container_width=True):
        with st.spinner("Processing documents..."):
            if replace_existing:
                reset_vectorstore()
            file_paths = _save_uploaded_files(uploaded_files or [])
            urls = [u.strip() for u in url_input.splitlines() if u.strip()]
            docs = load_documents(file_paths=file_paths, urls=urls)
            count = add_documents_to_vectorstore(docs)
        if count == 0:
            if file_paths or urls:
                st.warning(
                    "No usable content was added. Your files/URLs were received, but extracted text quality was too low. "
                    "Try a clearer PDF (searchable text), or export slides/notes to TXT/DOCX for best results."
                )
            else:
                st.warning("No content was added. Check your files/URLs and try again.")
        else:
            st.success(f"Added {count} chunks to ChromaDB.")

    if st.button("Clear Chat History", use_container_width=True):
        clear_messages()
        st.success("Chat history cleared.")

    if st.button("Reset Vector Store", use_container_width=True):
        reset_vectorstore()
        st.success("Vector store reset.")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("used_model"):
            st.caption(f"Using: {msg['used_model']}")
        if msg.get("sources"):
            st.caption("Source: " + ", ".join(msg["sources"]))

user_query = st.chat_input("Ask GH Buddy a question...")
if user_query:
    append_message("user", user_query)
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources, used_model = generate_answer(
                query=user_query,
                model_choice=st.session_state.model_choice,
                k=4,
            )
        st.markdown(answer)
        st.caption(f"Using: {used_model}")
        if sources:
            st.caption("Source: " + ", ".join(sources))

    append_message("assistant", answer, sources, used_model=used_model)
    log_chat(user_query, answer, st.session_state.model_choice, sources)

st.markdown("---")
st.caption("Built with Streamlit, ChromaDB, Gemini, and Hugging Face | GH Buddy Team")
