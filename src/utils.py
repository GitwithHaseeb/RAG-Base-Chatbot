"""Utilities for logging and Streamlit session state."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import List

import streamlit as st


LOG_FILE = Path("chat_logs.csv")


def init_session_state() -> None:
    """Initialize Streamlit session variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "model_choice" not in st.session_state:
        st.session_state.model_choice = "Gemini"


def append_message(role: str, content: str, sources: List[str] | None = None, used_model: str = "") -> None:
    """Add a chat message to session state."""
    if sources is None:
        sources = []
    st.session_state.messages.append({"role": role, "content": content, "sources": sources, "used_model": used_model})


def clear_messages() -> None:
    """Clear all conversation messages."""
    st.session_state.messages = []


def log_chat(query: str, response: str, model_name: str, sources: List[str]) -> None:
    """Append a query/response pair to CSV logs."""
    exists = LOG_FILE.exists()
    try:
        with LOG_FILE.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not exists:
                writer.writerow(["timestamp", "model", "query", "response", "sources"])
            writer.writerow([datetime.now().isoformat(), model_name, query, response, ";".join(sources)])
    except OSError:
        # Logging should never crash the user-facing app.
        return
