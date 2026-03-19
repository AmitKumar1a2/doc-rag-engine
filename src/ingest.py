from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

DATA_DIR = Path("data")


def load_pdf_documents(data_dir: Path = DATA_DIR) -> List[Document]:
    documents: List[Document] = []

    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {data_dir.resolve()}")

    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_file"] = pdf_file.name
        documents.extend(docs)

    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(documents)


def main() -> None:
    documents = load_pdf_documents()
    chunks = split_documents(documents)
    print(f"Loaded {len(documents)} pages and created {len(chunks)} chunks.")


if __name__ == "__main__":
    main()
