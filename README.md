# Doc RAG Engine

A fully local Retrieval-Augmented Generation (RAG) system for querying PDF documents.

## Overview

This project implements an end-to-end RAG pipeline using:

- **LangChain** – orchestration layer
- **FAISS** – vector similarity search
- **Ollama** – local embeddings and LLM inference

The system ingests PDFs, builds a vector index, and answers questions grounded in retrieved document context.

---

## Project Highlight

Designed and built a **fully local RAG system** with a focus on:

- Retrieval pipeline design (chunking, embedding, indexing)
- Separation of concerns between **embedding model** and **generation model**
- Context grounding and prompt structuring for reliable answers

AI-assisted development was used to accelerate implementation, while maintaining full ownership of architecture, debugging, and system design decisions.

---

## System Architecture

```text
PDFs → Chunking → Embeddings → FAISS Index → Retrieval → Prompt → LLM (Ollama)
```

## Python Version Requirement

This project requires **Python 3.11**.

Why:
- `LangChain` and related dependencies currently run more reliably on `3.11` in this setup.
- On `Python 3.14`, you may see compatibility warnings related to legacy Pydantic V1 internals from upstream libraries.
- Using `3.11` avoids those warnings and provides a stable baseline for this repo.

## Project Structure

```text
doc-rag-engine/
|-- data/
|   `-- AI-for-Education-RAG.pdf      # Source document(s) for ingestion
|-- vector_store/                     # Generated FAISS index files (created after indexing)
|-- src/
|   |-- ingest.py                     # Loads PDF pages and splits into chunks
|   |-- retriever.py                  # Builds and loads FAISS vector store
|   |-- rag_pipeline.py               # Retrieval + prompt construction + LLM response
|   `-- cli_chat.py                   # Interactive terminal chat loop
|-- requirements.txt                  # Python dependencies
`-- README.md
```

## File Documentation

### `src/ingest.py`
- Loads all PDFs from `data/`.
- Converts PDFs to LangChain `Document` objects.
- Splits documents into chunks using `RecursiveCharacterTextSplitter`.
- Can be run directly to verify ingestion/chunking counts.

### `src/retriever.py`
- Calls ingestion and chunking functions.
- Creates embeddings using `OllamaEmbeddings` (`nomic-embed-text`).
- Builds FAISS index and saves it under `vector_store/`.
- Exposes `load_vector_store()` for query-time retrieval.

### `src/rag_pipeline.py`
- Loads FAISS index from disk.
- Retrieves top-k relevant chunks for a user question.
- Formats context with chunk/source/page metadata.
- Calls `ChatOllama` (`llama3.2:3b`) to produce grounded answers.

### `src/cli_chat.py`
- CLI entry point for interactive Q&A.
- Takes user questions in a loop.
- Prints model answer and top source chunks/pages.

### `requirements.txt`
- Contains the dependency set for LangChain, FAISS, PDF parsing, environment handling, and progress utilities.

## Prerequisites

1. Install **Python 3.11**.
2. Install [Ollama](https://ollama.com/).
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

### 1) Ingest and index documents

```powershell
python src\retriever.py
```

Expected output:
- Number of indexed chunks
- FAISS files written to `vector_store/`

### 2) Start chat

```powershell
python src\cli_chat.py
```

Type questions in the terminal and use `exit` or `quit` to stop.

## Typical Workflow

1. Add or replace PDFs in `data/`.
2. Re-run indexing: `python src\retriever.py`.
3. Ask questions: `python src\cli_chat.py`.

## Notes

- Rebuild the vector store whenever source PDFs change.
- Ensure Ollama is running before indexing/chat.
- If retrieval quality is low, tune chunk size/overlap in `src/ingest.py`.
