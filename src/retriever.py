from functools import lru_cache
from pathlib import Path

from langchain_community.vectorstores import FAISS
from ingest import load_pdf_documents, split_documents
from model_config import get_embeddings

VECTOR_STORE_DIR = Path("vector_store")



def build_vector_store():
    documents = load_pdf_documents()
    chunks = split_documents(documents)

    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(VECTOR_STORE_DIR))
    load_vector_store.cache_clear()

    print(f"Indexed {len(chunks)} chunks into {VECTOR_STORE_DIR.resolve()}")


@lru_cache(maxsize=1)
def load_vector_store():
    vector_store = FAISS.load_local(
        str(VECTOR_STORE_DIR),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )
    return vector_store


if __name__ == "__main__":
    build_vector_store()
