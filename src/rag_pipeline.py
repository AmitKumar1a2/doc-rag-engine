from langchain_ollama import ChatOllama
from langchain_core.documents import Document

from retriever import load_vector_store


PROMPT_TEMPLATE = """
You are a helpful document assistant.

Use only the provided context to answer the question.
If the answer is not in the context, say clearly:
"I could not find that in the provided documents."

Context:
{context}

Question:
{question}

Answer:
""".strip()


def format_context(docs: list[Document]) -> str:
    formatted_parts = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", "unknown")
        content = doc.page_content.strip()

        formatted_parts.append(
            f"[Chunk {i} | Source: {source} | Page: {page}]\n{content}"
        )

    return "\n\n".join(formatted_parts)


def ask_question(question: str, k: int = 4):
    vector_store = load_vector_store()
    retrieved_docs = vector_store.similarity_search(question, k=k)

    context = format_context(retrieved_docs)

    llm = ChatOllama(model="llama3.2:3b", temperature=0.2) # Adjusted temperature to include more creativity while still being focused on the retrieved context. Make it higher (e.g., 0.5) for more creative answers, or lower (e.g., 0.1) for more deterministic answers.
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": retrieved_docs
    }
