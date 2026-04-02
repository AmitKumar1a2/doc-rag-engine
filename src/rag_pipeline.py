import json
import logging
import os
import re
import time
from typing import Any

from langchain_core.documents import Document

from model_config import (
    get_answer_llm,
    get_classifier_llm,
    get_conversation_llm,
    get_reranker_llm,
)
from retriever import load_vector_store

logger = logging.getLogger("rag_pipeline")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

NO_ANSWER_MESSAGE = "I could not find relevant information in the provided documents."

QUERY_TYPES = {"CONVERSATION", "FACT_LOOKUP", "SUMMARY"}
FACT_FINAL_K = 4

CLASSIFIER_PROMPT = """
You are a query router for a document assistant.

Task:
Classify the user query intent into exactly one label:
- CONVERSATION
- FACT_LOOKUP
- SUMMARY

Important:
- Classify intent only.
- Do NOT decide whether the answer exists in documents.
- Do NOT answer the user question.
- Return valid JSON only (no markdown, no extra text).

Return this schema exactly:`n{{`n  "query_type": "CONVERSATION|FACT_LOOKUP|SUMMARY",`n  "confidence": 0.0,`n  "reason": "short reason"`n}}

User query:
{question}
""".strip()

FACT_PROMPT_TEMPLATE = """
You are a strict document-grounded assistant.

Rules:
1. Use only the provided context.
2. Do not use outside knowledge.
3. Every factual claim must include citation tags like [1], [2].
4. Use only citation numbers that exist in the context block.
5. If context is insufficient, answer exactly:
"I could not find relevant information in the provided documents."

Context:
{context}

Question:
{question}

Answer:
""".strip()

SUMMARY_PROMPT_TEMPLATE = """
You are a document summarization assistant.

Rules:
1. Summarize only using the provided context.
2. Provide a concise synthesis across relevant chunks.
3. Add citations for key points where possible using [1], [2], ...
4. If context is insufficient, answer exactly:
"I could not find relevant information in the provided documents."

Context:
{context}

Question:
{question}

Answer:
""".strip()

CONVERSATION_PROMPT_TEMPLATE = """
You are a friendly local assistant in a document RAG app.

Rules:
1. This is conversational mode; no document retrieval is used.
2. Keep response brief and natural.
3. If user asks for document facts, suggest asking a specific document question.

User message:
{question}

Answer:
""".strip()

RERANK_PROMPT_TEMPLATE = """
You are a document reranker for factual question answering.

Task:
Select the best candidate passage IDs for answering the user's question.

Rules:
1. Prioritize passages that directly answer the question.
2. Prefer precise factual evidence over broad background.
3. Use only the candidate passages provided.
4. Return valid JSON only with no extra text.
5. Return exactly {top_k} IDs in best-to-worst order.

Return this schema exactly:
{{
  "top_ids": [3, 1, 2]
}}

Question:
{question}

Candidate passages:
{candidates}
""".strip()


VECTOR_STORE = load_vector_store()


def _safe_json_load(text: str) -> dict[str, Any] | None:
    text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # Robust fallback: extract first JSON object from noisy model output.
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None

    try:
        data = json.loads(match.group(0))
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        return None

    return None


def _validate_classifier_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    query_type = str(payload.get("query_type", "")).strip().upper()
    if query_type not in QUERY_TYPES:
        return None

    try:
        confidence = float(payload.get("confidence"))
    except (TypeError, ValueError):
        return None

    if confidence < 0.0 or confidence > 1.0:
        return None

    reason = str(payload.get("reason", "")).strip()
    if not reason:
        reason = "No reason provided."

    return {
        "query_type": query_type,
        "confidence": confidence,
        "reason": reason,
    }


def classify_query_rule_based(question: str) -> dict[str, Any]:
    q = question.strip().lower()

    conversation_keywords = {
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
        "how are you",
        "good morning",
        "good evening",
    }
    summary_keywords = {
        "summarize",
        "summary",
        "overview",
        "key points",
        "high level",
        "big picture",
        "main ideas",
    }

    if any(kw in q for kw in conversation_keywords):
        return {
            "query_type": "CONVERSATION",
            "confidence": 0.55,
            "reason": "Rule fallback: conversational greeting/thanks pattern.",
            "route_source": "RULE_FALLBACK",
        }

    if any(kw in q for kw in summary_keywords):
        return {
            "query_type": "SUMMARY",
            "confidence": 0.6,
            "reason": "Rule fallback: summary intent keywords detected.",
            "route_source": "RULE_FALLBACK",
        }

    return {
        "query_type": "FACT_LOOKUP",
        "confidence": 0.6,
        "reason": "Rule fallback: default to factual lookup intent.",
        "route_source": "RULE_FALLBACK",
    }


def classify_query(question: str) -> dict[str, Any]:
    prompt = CLASSIFIER_PROMPT.format(question=question)

    try:
        raw_response = get_classifier_llm().invoke(prompt).content
        payload = _safe_json_load(raw_response)
        if payload is None:
            return classify_query_rule_based(question)

        validated = _validate_classifier_payload(payload)
        if validated is None:
            return classify_query_rule_based(question)

        validated["route_source"] = "LLM_CLASSIFIER"
        return validated
    except Exception:
        return classify_query_rule_based(question)


def format_context(docs: list[Document]) -> str:
    formatted_parts: list[str] = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", "unknown")
        content = doc.page_content.strip()

        formatted_parts.append(f"[{i}] Source: {source} | Page: {page}\n{content}")

    return "\n\n".join(formatted_parts)


def format_rerank_candidates(docs: list[Document]) -> str:
    candidate_parts: list[str] = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", "unknown")
        content = doc.page_content.strip()
        truncated_content = content[:900]

        candidate_parts.append(
            f"[{i}] Source: {source} | Page: {page}\n{truncated_content}"
        )

    return "\n\n".join(candidate_parts)


def filter_by_score(
    docs_with_scores: list[tuple[Document, float]],
    score_threshold: float,
) -> list[Document]:
    # FAISS score is a distance metric in this setup (lower is better).
    filtered_docs: list[Document] = []
    for doc, score in docs_with_scores:
        if score <= score_threshold:
            filtered_docs.append(doc)
    return filtered_docs


def filter_with_scores(
    docs_with_scores: list[tuple[Document, float]],
    score_threshold: float,
) -> list[tuple[Document, float]]:
    # FAISS score is a distance metric in this setup (lower is better).
    filtered: list[tuple[Document, float]] = []
    for doc, score in docs_with_scores:
        if score <= score_threshold:
            filtered.append((doc, score))
    return filtered


def _get_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default

    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default

# Adaptive reranking logic to skip reranking when initial retrieval is confident and fast mode is enabled
def should_skip_rerank(
    filtered_docs_with_scores: list[tuple[Document, float]],
    *,
    final_k: int,
    fast_mode_enabled: bool,
    confident_gap_threshold: float,
) -> tuple[bool, str]:
    if not fast_mode_enabled:
        return False, "fast_mode_disabled"

    if len(filtered_docs_with_scores) <= 1:
        return True, "single_candidate"

    if len(filtered_docs_with_scores) <= final_k:
        return True, "already_small_candidate_set"

    top_score = filtered_docs_with_scores[0][1]
    next_score = filtered_docs_with_scores[1][1]
    score_gap = next_score - top_score
    if score_gap >= confident_gap_threshold:
        return True, "strong_top_hit"

    return False, "needs_rerank"


def build_sources_map(docs: list[Document]) -> dict[int, tuple[str, int | str]]:
    sources: dict[int, tuple[str, int | str]] = {}
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", "unknown")
        sources[i] = (source, page)
    return sources


def validate_citations(answer: str, sources: dict[int, tuple[str, int | str]]) -> bool:
    cited_numbers = set(map(int, re.findall(r"\[(\d+)\]", answer)))
    if not cited_numbers:
        return False

    valid_numbers = set(sources.keys())
    return cited_numbers.issubset(valid_numbers)


def filter_sources_to_citations(
    answer: str,
    sources: dict[int, tuple[str, int | str]],
) -> dict[int, tuple[str, int | str]]:
    cited_numbers = []
    seen_numbers: set[int] = set()

    for match in re.findall(r"\[(\d+)\]", answer):
        citation_id = int(match)
        if citation_id in sources and citation_id not in seen_numbers:
            cited_numbers.append(citation_id)
            seen_numbers.add(citation_id)

    return {citation_id: sources[citation_id] for citation_id in cited_numbers}


def rerank_fact_candidates(
    question: str,
    docs: list[Document],
    *,
    final_k: int,
) -> list[Document]:
    if len(docs) <= 1:
        return docs[:final_k]

    prompt = RERANK_PROMPT_TEMPLATE.format(
        question=question,
        candidates=format_rerank_candidates(docs),
        top_k=final_k,
    )

    try:
        raw_response = get_reranker_llm().invoke(prompt).content
        payload = _safe_json_load(raw_response)
        if payload is None:
            return docs[:final_k]

        top_ids = payload.get("top_ids")
        if not isinstance(top_ids, list):
            legacy_ids = payload.get("ranked_ids")
            top_ids = legacy_ids if isinstance(legacy_ids, list) else None
        if not isinstance(top_ids, list):
            return docs[:final_k]

        ranked_docs: list[Document] = []
        seen_ids: set[int] = set()
        for candidate_id in top_ids:
            if not isinstance(candidate_id, int):
                continue
            if candidate_id < 1 or candidate_id > len(docs):
                continue
            if candidate_id in seen_ids:
                continue

            ranked_docs.append(docs[candidate_id - 1])
            seen_ids.add(candidate_id)

            if len(ranked_docs) >= final_k:
                return ranked_docs

        if ranked_docs:
            for i, doc in enumerate(docs, start=1):
                if i not in seen_ids:
                    ranked_docs.append(doc)
                if len(ranked_docs) >= final_k:
                    break
            return ranked_docs
    except Exception:
        return docs[:final_k]

    return docs[:final_k]


def retrieve_by_similarity(
    question: str,
    *,
    k: int,
    score_threshold: float,
) -> list[Document]:
    docs_with_scores = VECTOR_STORE.similarity_search_with_score(question, k=k)
    return filter_by_score(docs_with_scores, score_threshold=score_threshold)


def retrieve_by_mmr(
    question: str,
    *,
    k: int,
    fetch_k: int,
    lambda_mult: float,
) -> list[Document]:
    return VECTOR_STORE.max_marginal_relevance_search(
        question,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
    )


def answer_with_retrieval(
    question: str,
    *,
    retrieved_docs: list[Document],
    prompt_template: str,
    strict_citation: bool,
) -> dict[str, Any]:
    if not retrieved_docs:
        return {
            "answer": NO_ANSWER_MESSAGE,
            "sources": {},
        }

    context = format_context(retrieved_docs)
    sources = build_sources_map(retrieved_docs)

    prompt = prompt_template.format(context=context, question=question)
    response = get_answer_llm().invoke(prompt)
    answer = response.content.strip()

    if strict_citation and not validate_citations(answer, sources):
        return {
            "answer": NO_ANSWER_MESSAGE,
            "sources": {},
        }

    if not strict_citation:
        # Relaxed mode: citations are encouraged, but if present they must still be valid.
        if re.search(r"\[(\d+)\]", answer) and not validate_citations(answer, sources):
            return {
                "answer": NO_ANSWER_MESSAGE,
                "sources": {},
            }

    cited_sources = filter_sources_to_citations(answer, sources)

    return {
        "answer": answer,
        "sources": cited_sources,
    }


def ask_question(question: str) -> dict[str, Any]:
    overall_start = time.perf_counter()
    routing_start = overall_start
    routing = classify_query(question)
    routing_time = time.perf_counter() - routing_start
    query_type = routing["query_type"]
    logger.info("routing took %.3fs -> %s", routing_time, query_type)

    if query_type == "CONVERSATION":
        conv_start = time.perf_counter()
        prompt = CONVERSATION_PROMPT_TEMPLATE.format(question=question)
        response = get_conversation_llm().invoke(prompt)
        logger.info("conversation route in %.3fs", time.perf_counter() - conv_start)
        logger.info("overall latency %.3fs", time.perf_counter() - overall_start)
        return {
            "answer": response.content.strip(),
            "sources": {},
            "query_type": query_type,
            "routing": routing,
        }

    if query_type == "FACT_LOOKUP":
        retrieval_start = time.perf_counter()
        docs_with_scores = VECTOR_STORE.similarity_search_with_score(question, k=8) # updating this to 8 to reduce latency and since we have a reranker step, we can be more aggressive with initial retrieval--
        filtered_docs_with_scores = filter_with_scores(
            docs_with_scores,
            score_threshold=1.4,
        )
        candidate_docs = [doc for doc, _ in filtered_docs_with_scores]
        logger.info(
            "fact retrieval (similarity) in %.3fs, candidates=%d",
            time.perf_counter() - retrieval_start,
            len(candidate_docs),
        )

        fast_mode_enabled = _get_bool_env("RAG_FACT_FAST_MODE", False)
        confident_gap_threshold = float(os.getenv("RAG_RERANK_GAP_THRESHOLD", "0.12"))
        skip_rerank, skip_reason = should_skip_rerank(
            filtered_docs_with_scores,
            final_k=FACT_FINAL_K,
            fast_mode_enabled=fast_mode_enabled,
            confident_gap_threshold=confident_gap_threshold,
        )

        if skip_rerank:
            retrieved_docs = candidate_docs[:FACT_FINAL_K]
            logger.info(
                "rerank skipped (%s), selected top-%d directly",
                skip_reason,
                len(retrieved_docs),
            )
        else:
            rerank_start = time.perf_counter()
            retrieved_docs = rerank_fact_candidates(
                question,
                candidate_docs,
                final_k=FACT_FINAL_K,
            )
            logger.info(
                "rerank step in %.3fs, final_k=%d",
                time.perf_counter() - rerank_start,
                len(retrieved_docs),
            )

        answer_start = time.perf_counter()
        result = answer_with_retrieval(
            question,
            retrieved_docs=retrieved_docs,
            prompt_template=FACT_PROMPT_TEMPLATE,
            strict_citation=True,
        )
        logger.info("answer step in %.3fs", time.perf_counter() - answer_start)
    else:  # SUMMARY
        retrieval_start = time.perf_counter()
        retrieved_docs = retrieve_by_mmr(
            question,
            k=8,
            fetch_k=20,
            lambda_mult=0.5,
        )
        logger.info(
            "summary retrieval (MMR) in %.3fs, docs=%d",
            time.perf_counter() - retrieval_start,
            len(retrieved_docs),
        )

        answer_start = time.perf_counter()
        result = answer_with_retrieval(
            question,
            retrieved_docs=retrieved_docs,
            prompt_template=SUMMARY_PROMPT_TEMPLATE,
            strict_citation=False,
        )
        logger.info("answer step in %.3fs", time.perf_counter() - answer_start)

    result["query_type"] = query_type
    result["routing"] = routing
    logger.info("overall latency %.3fs", time.perf_counter() - overall_start)
    return result
