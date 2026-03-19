from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

from ingest import load_pdf_documents, split_documents

VECTOR_STORE_DIR = Path("vector_store")


def build_vector_store():
    documents = load_pdf_documents()
    chunks = split_documents(documents)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = FAISS.from_documents(chunks, embeddings)

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(VECTOR_STORE_DIR))

    print(f"Indexed {len(chunks)} chunks into {VECTOR_STORE_DIR.resolve()}")


def load_vector_store():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = FAISS.load_local(
        str(VECTOR_STORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vector_store


if __name__ == "__main__":
    build_vector_store()
