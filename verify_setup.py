"""Quick validation script for GH Buddy setup."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent / "src"))


def check_sample_data() -> tuple[bool, list[str]]:
    required = ["data/syllabus.pdf", "data/python_notes.docx"]
    missing = [p for p in required if not os.path.exists(p) or os.path.getsize(p) == 0]
    return (len(missing) == 0, missing)


def warn_missing_keys() -> None:
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    hf_key = os.getenv("HUGGINGFACE_API_KEY", "").strip()
    if not gemini_key:
        print("[WARN] GEMINI_API_KEY is not set.")
    if not hf_key:
        print("[WARN] HUGGINGFACE_API_KEY is not set.")


def main() -> None:
    load_dotenv()
    print("== GH Buddy setup verification ==")
    ok, missing = check_sample_data()
    if not ok:
        print(f"[FAIL] Missing or empty sample files: {missing}")
        raise SystemExit(1)
    print("[OK] Sample files found.")

    try:
        from load_documents import load_documents
    except ModuleNotFoundError as exc:
        print(f"[WARN] Document loading test skipped due to missing dependency: {exc}")
        print("[WARN] Run `pip install -r requirements.txt` in Python 3.10/3.11 and rerun.")
        print("\n[OK] Verification finished with dependency warnings.")
        return

    docs = load_documents(file_paths=["data/syllabus.pdf", "data/python_notes.docx"])
    if not docs:
        print("[FAIL] Document loading returned empty content.")
        raise SystemExit(1)
    print(f"[OK] Loaded {len(docs)} document entries.")

    try:
        from create_vectorstore import add_documents_to_vectorstore, reset_vectorstore

        reset_vectorstore()
        chunk_count = add_documents_to_vectorstore(docs)
        if chunk_count <= 0:
            print("[FAIL] No chunks added to vector store.")
            raise SystemExit(1)
        print(f"[OK] Added {chunk_count} chunks to vector store.")
    except ModuleNotFoundError as exc:
        print(f"[WARN] Vector store test skipped due to missing dependency: {exc}")
        print("[WARN] Run `pip install -r requirements.txt` in a Python 3.10/3.11 environment.")
        chunk_count = 0

    warn_missing_keys()
    query = "What is covered in week 2?"

    try:
        from rag_chain import generate_answer

        gemini_answer, gemini_sources, gemini_model = generate_answer(query=query, model_choice="Gemini", k=4)
        print("\n[Gemini] Answer:", gemini_answer[:220])
        print("[Gemini] Model:", gemini_model)
        print("[Gemini] Sources:", gemini_sources)

        hf_answer, hf_sources, hf_model = generate_answer(query=query, model_choice="Hugging Face", k=4)
        print("\n[Hugging Face] Answer:", hf_answer[:220])
        print("[Hugging Face] Model:", hf_model)
        print("[Hugging Face] Sources:", hf_sources)
    except ModuleNotFoundError as exc:
        print(f"[WARN] Model smoke test skipped due to missing dependency: {exc}")
        print("[WARN] Install full requirements first.")

    print("\n[OK] Verification finished.")


if __name__ == "__main__":
    main()
