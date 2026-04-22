"""RAG prompt and model inference layer for GH Buddy."""

from __future__ import annotations

import os
import re
from typing import Dict, List, Tuple

import requests
from dotenv import load_dotenv
try:
    from google import genai as genai_new
except ModuleNotFoundError:
    genai_new = None

from create_vectorstore import retrieve_many_for_summary, retrieve_top_k


load_dotenv()

PROMPT_TEMPLATE = """You are a helpful student assistant named GH Buddy. Use the following context to answer the student's question. If the answer is not in the context, say "I don't know, but I can ask your teacher for you." Be concise and friendly.
Reply in {response_language}.
If the student asks to explain, teach in simple classroom style with examples.
Context: {context}
Question: {query}
Answer:
"""

GENERAL_PROMPT_TEMPLATE = """You are a helpful student assistant named GH Buddy. Be concise, friendly, and accurate.
Reply in {response_language}.
If user says they missed class or did not understand, respond warmly and explain step-by-step in simple words.
Question: {query}
Answer:
"""

EXPLAIN_PROMPT_TEMPLATE = """You are GH Buddy, a student-friendly tutor.
Reply in {response_language}.
Use ONLY provided context. Explain in simple teaching style for a student who missed class.

Requirements:
- Start with a short reassuring line.
- Explain core idea first, then details.
- Use simple examples where possible.
- Keep topic-focused and avoid irrelevant details.
- End with a quick revision checklist (3-5 bullets).

Context:
{context}

Question:
{query}

Answer:
"""

SUMMARY_PROMPT_TEMPLATE = """You are GH Buddy, a student exam-prep assistant.
Use ONLY the provided context and produce a comprehensive, structured summary.
Reply in {response_language}.

Requirements:
- Cover all major topics and subtopics present in context.
- Use the EXACT section format below with Markdown headings.
- Include key definitions, models, frameworks, and process steps.
- Include exam-focused takeaways and likely scenario/application areas.
- Do not skip important sections; do not be overly brief.
- If some part is unclear, say so explicitly.

Output format (follow exactly):
## 1) Definitions
- Key terms and meanings

## 2) Models and Figures
- Important models/diagrams and what each explains

## 3) Core Factors and Concepts
- Main conceptual buckets and critical details

## 4) Processes and Steps
- Any stepwise frameworks in correct order

## 5) Scenario-Based Exam Angles
- Practical application-style question themes and what examiner expects

## 6) Quick Revision Bullets
- 10 concise bullets for last-minute revision

Context:
{context}

Question:
{query}

Answer:
"""


def retrieve_context(query: str, k: int = 4) -> List[Dict[str, str]]:
    """Return top-k context chunks for the query."""
    try:
        if _is_summary_query(query):
            return retrieve_many_for_summary(query, k=max(12, k))
        return retrieve_top_k(query, k=k)
    except Exception:
        return []


def _build_prompt(query: str, context_chunks: List[Dict[str, str]], response_language: str) -> str:
    context = "\n\n".join(chunk["content"] for chunk in context_chunks) or "No context found."
    return PROMPT_TEMPLATE.format(context=context, query=query, response_language=response_language)


def _build_general_prompt(query: str, response_language: str) -> str:
    return GENERAL_PROMPT_TEMPLATE.format(query=query, response_language=response_language)


def _build_summary_prompt(query: str, context_chunks: List[Dict[str, str]], response_language: str) -> str:
    context = "\n\n".join(chunk["content"] for chunk in context_chunks) or "No context found."
    return SUMMARY_PROMPT_TEMPLATE.format(context=context, query=query, response_language=response_language)


def _build_explain_prompt(query: str, context_chunks: List[Dict[str, str]], response_language: str) -> str:
    context = "\n\n".join(chunk["content"] for chunk in context_chunks) or "No context found."
    return EXPLAIN_PROMPT_TEMPLATE.format(context=context, query=query, response_language=response_language)


def _extract_candidate_points(context_chunks: List[Dict[str, str]], limit: int = 20) -> List[str]:
    points: List[str] = []
    for chunk in context_chunks:
        text = chunk.get("content", "")
        for raw in re.split(r"[.\n]", text):
            s = " ".join(raw.split()).strip(" -:;")
            if len(s) >= 28 and s not in points:
                points.append(s)
            if len(points) >= limit:
                return points
    return points


def _enforce_summary_structure(answer: str, context_chunks: List[Dict[str, str]]) -> str:
    """Ensure final summary always contains all fixed sections and 10 revision bullets."""
    required_headings = [
        "## 1) Definitions",
        "## 2) Models and Figures",
        "## 3) Core Factors and Concepts",
        "## 4) Processes and Steps",
        "## 5) Scenario-Based Exam Angles",
        "## 6) Quick Revision Bullets",
    ]

    out = answer.strip()
    for heading in required_headings:
        if heading not in out:
            out += f"\n\n{heading}\n- Not clearly covered in extracted text."

    # Guarantee 10 bullets in section 6.
    lines = out.splitlines()
    section_start = None
    for i, line in enumerate(lines):
        if line.strip() == "## 6) Quick Revision Bullets":
            section_start = i
            break

    if section_start is not None:
        section_end = len(lines)
        for j in range(section_start + 1, len(lines)):
            if lines[j].startswith("## "):
                section_end = j
                break
        bullet_count = sum(1 for ln in lines[section_start + 1 : section_end] if ln.strip().startswith("- "))
        needed = max(0, 10 - bullet_count)
        if needed > 0:
            candidates = _extract_candidate_points(context_chunks, limit=20)
            if not candidates:
                candidates = [
                    "Revise the core definitions and formulas from this lecture.",
                    "Practice scenario-based application of each concept.",
                    "Focus on models, their inputs, and expected outputs.",
                    "Compare related concepts to avoid confusion in exams.",
                    "Memorize step-by-step processes in correct order.",
                ]
            additions = [f"- {candidates[idx % len(candidates)]}" for idx in range(needed)]
            lines = lines[:section_end] + additions + lines[section_end:]
            out = "\n".join(lines)

    return out.strip()


def _fallback_response() -> str:
    return "Please set your API keys."


def _is_greeting(query: str) -> bool:
    q = query.strip().lower()
    return bool(re.fullmatch(r"(hi|hello|hey|salam|assalam o alaikum|assalamualaikum)", q))


def _detect_response_language(query: str) -> str:
    """Detect preferred response language from user query."""
    if re.search(r"[\u0600-\u06FF]", query):
        return "Urdu"
    q = query.lower()
    if "urdu" in q or "اردو" in q:
        return "Urdu"
    return "English"


def _is_summary_query(query: str) -> bool:
    q = query.lower()
    summary_terms = [
        "summarize",
        "summary",
        "exam-ready",
        "exam ready",
        "explain complete",
        "full explanation",
        "detailed explanation",
        "lecture summary",
        "khulasa",
        "tafseel",
        "explain in urdu",
        "سمری",
        "خلاصہ",
        "تفصیل",
        "سمجھا دو",
        "samjha do",
        "samjhao",
        "samjha do",
        "paper hai",
        "kal paper",
        "exam",
        "paper ki tayari",
        "important points",
        "main points",
    ]
    return any(term in q for term in summary_terms)


def _is_explain_query(query: str) -> bool:
    q = query.lower()
    explain_terms = [
        "explain",
        "samjha",
        "samjhao",
        "سمجھ",
        "understand",
        "didn't understand",
        "missed class",
        "lecture samjha do",
        "topic explain",
    ]
    return any(term in q for term in explain_terms)


def _intro_message() -> str:
    return (
        "Hi, I am your AI assistant. I will help you with your notes, labels, and URLs. "
        "For any subject, I will give concise, topic-focused answers."
    )


def _intro_message_urdu() -> str:
    return (
        "اسلام علیکم! میں آپ کا AI اسسٹنٹ ہوں۔ میں آپ کی لیکچرز، نوٹس اور URLs "
        "کی بنیاد پر سادہ اور امتحان کے لحاظ سے مفید وضاحت فراہم کروں گا۔"
    )


def _no_context_message() -> str:
    return (
        "I could not find indexed study material yet. "
        "Please click 'Process & Add to Vector Store' after uploading your PDF/DOCX/TXT, then ask again."
    )


def _no_context_message_urdu() -> str:
    return (
        "ابھی انڈیکس شدہ اسٹڈی میٹریل نہیں ملا۔ براہِ کرم PDF/DOCX/PPTX/TXT اپلوڈ کرنے کے بعد "
        "'Process & Add to Vector Store' پر کلک کریں، پھر دوبارہ سوال کریں۔"
    )


def _local_context_fallback(context_chunks: List[Dict[str, str]], response_language: str) -> str:
    """Return a best-effort local answer when remote APIs are unavailable."""
    if not context_chunks:
        return _no_context_message_urdu() if response_language == "Urdu" else _no_context_message()
    best = context_chunks[0].get("content", "").strip()
    if not best:
        return _no_context_message_urdu() if response_language == "Urdu" else _no_context_message()
    short = best.replace("\n", " ").strip()
    if len(short) > 260:
        short = short[:260].rstrip() + "..."
    if response_language == "Urdu":
        return f"آپ کے نوٹس سے فوری خلاصہ: {short}"
    return f"Quick answer from your notes: {short}"


def call_gemini(prompt: str) -> str:
    """Call Google Gemini with generated prompt."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your_key_here":
        return _fallback_response()

    try:
        if genai_new is None:
            return "Gemini SDK not available in this environment."

        client = genai_new.Client(api_key=api_key)
        candidate_models = [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-1.5-flash",
        ]
        last_error = None
        for model_name in candidate_models:
            try:
                response = client.models.generate_content(model=model_name, contents=prompt)
                text = (getattr(response, "text", "") or "").strip()
                if text:
                    return text
            except Exception as model_exc:
                last_error = model_exc
                continue
        if last_error is not None:
            raise last_error
        return "I couldn't generate a Gemini response."
    except Exception as exc:
        message = str(exc).lower()
        if "quota" in message or "429" in message or "resource_exhausted" in message:
            return "Gemini quota exceeded right now. Please retry later or switch to Hugging Face."
        if "api key" in message or "permission" in message or "unauthorized" in message:
            return "Gemini API key is invalid or missing permissions."
        return f"Gemini request failed: {str(exc)[:180]}"


def call_huggingface(prompt: str) -> str:
    """Call Hugging Face router chat-completions API with generated prompt."""
    api_key = os.getenv("HUGGINGFACE_API_KEY", "").strip()
    if not api_key or api_key == "your_key_here":
        return _fallback_response()

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    candidate_models = [
        "Qwen/Qwen2.5-7B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ]

    last_error = ""
    endpoint = "https://router.huggingface.co/v1/chat/completions"
    for model_name in candidate_models:
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1200,
            "temperature": 0.2,
        }
        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=60)
            if response.status_code == 401:
                return "Hugging Face API key is invalid."
            if response.status_code == 429:
                return "Hugging Face rate limit reached. Please retry later or switch to Gemini."
            if response.status_code in (400, 404):
                last_error = f"{model_name}: {response.text[:120]}"
                continue

            response.raise_for_status()
            data = response.json()
            choices = data.get("choices", []) if isinstance(data, dict) else []
            if choices:
                content = choices[0].get("message", {}).get("content", "")
                content = (content or "").strip()
                if content:
                    return content
            last_error = f"{model_name}: empty response"
        except requests.RequestException as exc:
            last_error = str(exc)[:180]
            continue

    return f"Hugging Face request failed: {last_error or 'No available model endpoint responded successfully.'}"


def generate_answer(query: str, model_choice: str = "Gemini", k: int = 4) -> Tuple[str, List[str], str]:
    """Retrieve context, build prompt, and call selected model."""
    response_language = _detect_response_language(query)

    if _is_greeting(query):
        if response_language == "Urdu":
            return _intro_message_urdu(), [], "System"
        return _intro_message(), [], "System"

    if _is_summary_query(query):
        effective_k = 28
    elif _is_explain_query(query):
        effective_k = max(k, 14)
    else:
        effective_k = k
    effective_k = max(effective_k, 1)

    context_chunks = retrieve_context(query, k=effective_k)
    if context_chunks and _is_summary_query(query):
        prompt = _build_summary_prompt(query, context_chunks, response_language=response_language)
    elif context_chunks and _is_explain_query(query):
        prompt = _build_explain_prompt(query, context_chunks, response_language=response_language)
    else:
        prompt = (
            _build_prompt(query, context_chunks, response_language=response_language)
            if context_chunks
            else _build_general_prompt(query, response_language=response_language)
        )
    sources = sorted({chunk["source"] for chunk in context_chunks}) if context_chunks else []

    if model_choice.lower().startswith("hugging"):
        answer = call_huggingface(prompt)
        used_model = "Hugging Face"
        if answer.lower().startswith("hugging face request failed"):
            answer = _local_context_fallback(context_chunks, response_language=response_language)
            used_model = "Local fallback"
    else:
        answer = call_gemini(prompt)
        used_model = "Gemini"
        if "quota exceeded" in answer.lower() or "resource_exhausted" in answer.lower():
            hf_answer = call_huggingface(prompt)
            if hf_answer and "please set your api keys" not in hf_answer.lower():
                if hf_answer.lower().startswith("hugging face request failed"):
                    answer = (
                        "Gemini quota hit, Hugging Face also unavailable. "
                        "Using local context fallback.\n\n"
                        + _local_context_fallback(context_chunks, response_language=response_language)
                    )
                    used_model = "Local fallback"
                else:
                    answer = f"Gemini quota hit, switched to Hugging Face automatically.\n\n{hf_answer}"
                    used_model = "HF fallback"
        elif not context_chunks and answer.strip():
            answer = f"{answer}\n\nNote: No indexed document context found yet."

    if _is_summary_query(query):
        answer = _enforce_summary_structure(answer, context_chunks)

    return answer, sources, used_model
