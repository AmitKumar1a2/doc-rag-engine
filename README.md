# Doc RAG Engine

A fully local Retrieval-Augmented Generation (RAG) system for querying PDF documents.

## Overview

This project implements an end-to-end RAG pipeline using:

- **LangChain** for orchestration
- **FAISS** for vector similarity search
- **Ollama** for local embeddings and LLM inference

The system ingests PDFs, builds a vector index, and answers questions grounded in retrieved document context.

## Project Highlight

Designed and built a fully local RAG system with focus on retrieval quality, explainability, and practical reliability.

AI-assisted development was used to accelerate implementation while maintaining full ownership of architecture, debugging, and design decisions.

## Recent Upgrades

The pipeline was upgraded with minimal, production-style changes:

- Citation-aware answers enforced via prompt (`[1]`, `[2]`, ...)
- Retrieval confidence filtering using `similarity_search_with_score`
- No-answer fallback when retrieval quality is weak
- Structured source mapping in output
- CLI updated to print citation mappings clearly

## System Architecture

```text
Indexing:
PDFs -> Chunking -> Embeddings -> FAISS Index

Query time:
Question -> Similarity Search With Score -> Threshold Filter -> Prompt With Context -> LLM
```

## Python Version Requirement

This project requires **Python 3.11**.

Why:
- `LangChain` and related dependencies are more stable in this setup on `3.11`.
- On `Python 3.14`, upstream compatibility warnings may appear around legacy Pydantic V1 internals.
- `3.11` gives a cleaner and more reliable baseline for local development.

## Project Structure

```text
doc-rag-engine/
|-- data/
|   `-- AI-for-Education-RAG.pdf      # Source document(s) for ingestion
|-- vector_store/                     # Generated FAISS index files
|-- src/
|   |-- ingest.py                     # PDF loading + chunking
|   |-- retriever.py                  # FAISS build/load
|   |-- rag_pipeline.py               # Retrieval filtering + prompt + answer generation
|   `-- cli_chat.py                   # Interactive CLI
|-- requirements.txt                  # Python dependencies
`-- README.md
```

## File Documentation

### `src/ingest.py`
- Loads PDFs from `data/`
- Converts pages to LangChain `Document` objects
- Splits documents with `RecursiveCharacterTextSplitter`

### `src/retriever.py`
- Builds FAISS from chunked documents
- Uses `nomic-embed-text` via `OllamaEmbeddings`
- Saves index to `vector_store/`
- Loads index for query-time retrieval

### `src/rag_pipeline.py`
- Uses `similarity_search_with_score`
- Filters low-confidence matches by threshold
- Enforces strict, citation-aware prompt behavior
- Returns structured output:

```python
{
  "answer": "...",
  "sources": {
    1: ("file.pdf", page),
    2: ("file.pdf", page)
  }
}
```

### `src/cli_chat.py`
- Runs interactive terminal Q&A
- Displays assistant answer
- Displays citations as `[id] file | page`
- Shows `(none)` when no relevant sources are found

## Prerequisites

1. Install **Python 3.11**
2. Install [Ollama](https://ollama.com/)
3. Pull required models:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

## Setup

From project root:

```powershell
py -3.11 -m venv rag_env311
.\rag_env311\Scripts\activate
python -m pip install -r requirements.txt
```

## How To Use

### 1) Build/refresh vector index

```powershell
python src\retriever.py
```

### 2) Start CLI chat

```powershell
python src\cli_chat.py
```

Type your question and use `exit` or `quit` to stop.

## Typical Workflow

1. Add or update PDFs in `data/`
2. Re-index with `python src\retriever.py`
3. Query with `python src\cli_chat.py`

## Notes

- Rebuild vector store whenever source PDFs change
- Keep Ollama running while indexing/chatting
- Tune chunk size or score threshold to balance recall and precision
