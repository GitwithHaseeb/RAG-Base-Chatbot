"""Create and query the ChromaDB vector store.

Falls back to a lightweight local store when optional deps are missing.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Dict, List
from uuid import uuid4

try:
    import chromadb
except ModuleNotFoundError:
    chromadb = None

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ModuleNotFoundError:
    RecursiveCharacterTextSplitter = None

try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError:
    SentenceTransformer = None


PERSIST_DIR = Path("./chroma_db")
COLLECTION_NAME = "gh_buddy_docs"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FALLBACK_STORE_PATH = PERSIST_DIR / "fallback_store.json"


def _has_chromadb() -> bool:
    return chromadb is not None and SentenceTransformer is not None and RecursiveCharacterTextSplitter is not None


def _get_collection():
    if not _has_chromadb():
        raise RuntimeError("ChromaDB stack unavailable.")
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    return client.get_or_create_collection(name=COLLECTION_NAME)


@lru_cache(maxsize=1)
def _get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def _split_text_simple(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    chunks: List[str] = []
    if not text.strip():
        return chunks
    step = max(1, chunk_size - overlap)
    for start in range(0, len(text), step):
        part = text[start : start + chunk_size].strip()
        if part:
            chunks.append(part)
    return chunks


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def _keyword_score(query: str, doc_text: str) -> float:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0
    doc_tokens = _tokenize(doc_text)
    overlap = len(query_tokens.intersection(doc_tokens))
    return overlap / max(1, len(query_tokens))


def _load_fallback_rows() -> List[Dict[str, str]]:
    if not FALLBACK_STORE_PATH.exists():
        return []
    try:
        return json.loads(FALLBACK_STORE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _save_fallback_rows(rows: List[Dict[str, str]]) -> None:
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    FALLBACK_STORE_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def chunk_documents(documents: List[Dict[str, str]], chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, str]]:
    """Split loaded documents into chunks with source metadata."""
    splitter = None
    if RecursiveCharacterTextSplitter is not None:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks: List[Dict[str, str]] = []

    for doc in documents:
        if splitter is not None:
            split_texts = splitter.split_text(doc["content"])
        else:
            split_texts = _split_text_simple(doc["content"], chunk_size=chunk_size, overlap=overlap)
        for idx, text in enumerate(split_texts):
            if text.strip():
                chunks.append(
                    {
                        "content": text,
                        "source": doc["source"],
                        "chunk_id": f"{doc['source']}-{idx}",
                    }
                )
    return chunks


def add_documents_to_vectorstore(documents: List[Dict[str, str]]) -> int:
    """Embed and persist document chunks in ChromaDB."""
    if not documents:
        return 0

    chunks = chunk_documents(documents)
    if not chunks:
        return 0

    if _has_chromadb():
        collection = _get_collection()
        model = _get_embedding_model()
        texts = [c["content"] for c in chunks]
        metadatas = [{"source": c["source"], "chunk_id": c["chunk_id"]} for c in chunks]
        ids = [str(uuid4()) for _ in chunks]
        embeddings = model.encode(texts, convert_to_numpy=True).tolist()
        collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    else:
        rows = _load_fallback_rows()
        rows.extend(chunks)
        _save_fallback_rows(rows)
    return len(chunks)


def retrieve_top_k(query: str, k: int = 4) -> List[Dict[str, str]]:
    """Retrieve top-k chunks for a query."""
    if _has_chromadb():
        collection = _get_collection()
        collection_count = collection.count()
        if collection_count == 0:
            return []
        model = _get_embedding_model()
        query_embedding = model.encode([query], convert_to_numpy=True).tolist()[0]
        result = collection.query(query_embeddings=[query_embedding], n_results=min(k, collection_count))
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        matches: List[Dict[str, str]] = []
        for doc, metadata in zip(documents, metadatas):
            matches.append({"content": doc, "source": metadata.get("source", "unknown")})
        return matches

    rows = _load_fallback_rows()
    if not rows:
        return []
    ranked = sorted(rows, key=lambda r: _keyword_score(query, r.get("content", "")), reverse=True)
    top_rows = ranked[: max(1, k)]
    return [{"content": r.get("content", ""), "source": r.get("source", "unknown")} for r in top_rows if r.get("content")]


def retrieve_many_for_summary(query: str, k: int = 24) -> List[Dict[str, str]]:
    """Retrieve a broader set of chunks for complete lecture summaries."""
    if _has_chromadb():
        collection = _get_collection()
        collection_count = collection.count()
        if collection_count == 0:
            return []
        model = _get_embedding_model()
        query_embedding = model.encode([query], convert_to_numpy=True).tolist()[0]
        result = collection.query(query_embeddings=[query_embedding], n_results=min(k, collection_count))
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        matches: List[Dict[str, str]] = []
        seen = set()
        for doc, metadata in zip(documents, metadatas):
            source = metadata.get("source", "unknown")
            key = (source, doc[:80])
            if key in seen:
                continue
            seen.add(key)
            matches.append({"content": doc, "source": source})
        return matches

    rows = _load_fallback_rows()
    if not rows:
        return []
    ranked = sorted(rows, key=lambda r: _keyword_score(query, r.get("content", "")), reverse=True)
    top_rows = ranked[: max(1, k)]
    return [{"content": r.get("content", ""), "source": r.get("source", "unknown")} for r in top_rows if r.get("content")]


def reset_vectorstore() -> None:
    """Delete persisted ChromaDB directory."""
    if PERSIST_DIR.exists():
        shutil.rmtree(PERSIST_DIR)
