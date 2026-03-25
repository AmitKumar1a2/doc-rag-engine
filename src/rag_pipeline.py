import json
import re
from typing import Any

from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from retriever import load_vector_store

NO_ANSWER_MESSAGE = "I could not find relevant information in the provided documents."

QUERY_TYPES = {"CONVERSATION", "FACT_LOOKUP", "SUMMARY"}

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
Rank the candidate passages from best to worst for answering the user's question.

Rules:
1. Prioritize passages that directly answer the question.
2. Prefer precise factual evidence over broad background.
3. Use only the candidate passages provided.
4. Return valid JSON only.

Return this schema exactly:
{{
  "ranked_ids": [3, 1, 2]
}}

Question:
{question}

Candidate passages:
{candidates}
""".strip()


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
    classifier_llm = ChatOllama(model="phi3:mini", temperature=0)
    prompt = CLASSIFIER_PROMPT.format(question=question)

    try:
        raw_response = classifier_llm.invoke(prompt).content
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


def rerank_fact_candidates(
    question: str,
    docs: list[Document],
    *,
    final_k: int,
) -> list[Document]:
    if len(docs) <= 1:
        return docs[:final_k]

    reranker_llm = ChatOllama(model="phi3:mini", temperature=0)
    prompt = RERANK_PROMPT_TEMPLATE.format(
        question=question,
        candidates=format_rerank_candidates(docs),
    )

    try:
        raw_response = reranker_llm.invoke(prompt).content
        payload = _safe_json_load(raw_response)
        if payload is None:
            return docs[:final_k]

        ranked_ids = payload.get("ranked_ids")
        if not isinstance(ranked_ids, list):
            return docs[:final_k]

        ranked_docs: list[Document] = []
        seen_ids: set[int] = set()
        for candidate_id in ranked_ids:
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
    vector_store = load_vector_store()
    docs_with_scores = vector_store.similarity_search_with_score(question, k=k)
    return filter_by_score(docs_with_scores, score_threshold=score_threshold)


def retrieve_by_mmr(
    question: str,
    *,
    k: int,
    fetch_k: int,
    lambda_mult: float,
) -> list[Document]:
    vector_store = load_vector_store()
    return vector_store.max_marginal_relevance_search(
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

    llm = ChatOllama(model="llama3.2:3b", temperature=0)
    prompt = prompt_template.format(context=context, question=question)
    response = llm.invoke(prompt)
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

    return {
        "answer": answer,
        "sources": sources,
    }


def ask_question(question: str) -> dict[str, Any]:
    routing = classify_query(question)
    query_type = routing["query_type"]

    if query_type == "CONVERSATION":
        llm = ChatOllama(model="llama3.2:3b", temperature=0.2)
        prompt = CONVERSATION_PROMPT_TEMPLATE.format(question=question)
        response = llm.invoke(prompt)
        return {
            "answer": response.content.strip(),
            "sources": {},
            "query_type": query_type,
            "routing": routing,
        }

    if query_type == "FACT_LOOKUP":
        candidate_docs = retrieve_by_similarity(
            question,
            k=12,
            score_threshold=1.4,
        )
        retrieved_docs = rerank_fact_candidates(
            question,
            candidate_docs,
            final_k=4,
        )
        result = answer_with_retrieval(
            question,
            retrieved_docs=retrieved_docs,
            prompt_template=FACT_PROMPT_TEMPLATE,
            strict_citation=True,
        )
    else:  # SUMMARY
        retrieved_docs = retrieve_by_mmr(
            question,
            k=8,
            fetch_k=20,
            lambda_mult=0.5,
        )
        result = answer_with_retrieval(
            question,
            retrieved_docs=retrieved_docs,
            prompt_template=SUMMARY_PROMPT_TEMPLATE,
            strict_citation=False,
        )

    result["query_type"] = query_type
    result["routing"] = routing
    return result
