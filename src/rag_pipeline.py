from langchain_ollama import ChatOllama
from langchain_core.documents import Document
import re
from retriever import load_vector_store

NO_ANSWER_MESSAGE = "I could not find relevant information in the provided documents. Please refine your query."

PROMPT_TEMPLATE = """
You are a strict document-grounded assistant.

Rules:
1. Use only the provided context.
2. Do not use outside knowledge.
3. Every factual claim must include citation tags like [1], [2].
    -Keep citations only as [1], [2], do not include source names or page numbers.
4. Use only citation numbers that exist in the context block.
5. If the context is insufficient, answer exactly:
"I could not find relevant information in the provided documents. Please refine your query."

Context:
{context}

Question:
{question}

Answer:
""".strip()



def validate_citations(answer: str, sources: dict):
    cited_numbers = set(map(int, re.findall(r"\[(\d+)\]", answer)))

    if not cited_numbers:
        return False

    valid_numbers = set(sources.keys())

    return cited_numbers.issubset(valid_numbers)

def format_context(docs: list[Document]) -> str:
    formatted_parts = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", "unknown")
        content = doc.page_content.strip()

        formatted_parts.append(
            f"[{i}] Source: {source} | Page: {page}\n{content}"
        )

    return "\n\n".join(formatted_parts)


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


def ask_question(question: str, k: int = 4, score_threshold: float = 1.2):
    vector_store = load_vector_store()
    docs_with_scores = vector_store.similarity_search_with_score(question, k=k)
    retrieved_docs = filter_by_score(docs_with_scores, score_threshold=score_threshold)

    if not retrieved_docs:
        return {
            "answer": NO_ANSWER_MESSAGE,
            "sources": {},
        }
    
    context = format_context(retrieved_docs)
    sources = build_sources_map(retrieved_docs)

    llm = ChatOllama(model="llama3.2:3b", temperature=0)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    response = llm.invoke(prompt)
# Validate that the retrieved documents can support a properly grounded answer before returning the response. If the answer contains citations that do not match the retrieved documents, return a message indicating that a properly grounded answer could not be generated.
    if not validate_citations(response.content, sources):
        return {
            "answer": "I could not generate a properly grounded answer from the provided documents.",
            "sources": {}
        }
    return {
        "answer": response.content,
        "sources": sources,
    }
