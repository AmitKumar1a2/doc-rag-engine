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

- Query routing for `CONVERSATION`, `FACT_LOOKUP`, and `SUMMARY`
- Citation-aware answers enforced via prompt (`[1]`, `[2]`, ...)
- Retrieval confidence filtering using `similarity_search_with_score`
- LLM-based reranking for fact lookup to improve final chunk precision
- MMR-based retrieval for summary questions to improve context diversity
- No-answer fallback when retrieval quality is weak
- Structured source mapping in output
- Stage-wise model configuration through environment variables
- CLI updated to print routing metadata and citation mappings clearly
- Streamlit UI added for PDF upload, querying, and answer history

## System Architecture

```text
Indexing:
PDFs -> Chunking -> Embeddings -> FAISS Index

Query time:
Question -> Query Router -> Route-Specific Retrieval / Conversation -> Prompt With Context -> LLM

Retrieval routes:
- `FACT_LOOKUP` -> Similarity Search With Score -> Threshold Filter -> LLM Reranking
- `SUMMARY` -> MMR Retrieval (`fetch_k` candidate pool -> diversified `k` results)
- `CONVERSATION` -> No retrieval
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
|-- docs/
|   |-- daily/                        # Daily conceptual learning notes
|   |-- RAG_Challenges_and_Solutions.md
|   `-- RAG_QA_Knowledge_Base.md
|-- vector_store/                     # Generated FAISS index files
|-- src/
|   |-- ingest.py                     # PDF loading + chunking
|   |-- model_config.py               # Stage-wise model and embedding configuration
|   |-- retriever.py                  # FAISS build/load
|   |-- rag_pipeline.py               # Query routing + retrieval + prompt + answer generation
|   `-- cli_chat.py                   # Interactive CLI
|-- streamlit_app.py                  # Streamlit UI for upload + query + history
|-- AGENTS.md                         # Repo-specific agent workflow instructions
|-- requirements.txt                  # Python dependencies
`-- README.md
```

## File Documentation

### `src/ingest.py`
- Loads PDFs from `data/`
- Converts pages to LangChain `Document` objects
- Splits documents with `RecursiveCharacterTextSplitter`

### `src/model_config.py`
- Centralizes model selection for classifier, reranker, answer, conversation, and embeddings
- Reads stage-specific model names and temperatures from environment variables
- Builds cached `ChatOllama` / `OllamaEmbeddings` clients so each stage can be swapped independently

### `src/retriever.py`
- Builds FAISS from chunked documents
- Uses the configured embedding model via `OllamaEmbeddings`
- Saves index to `vector_store/`
- Loads index for query-time retrieval

### `src/rag_pipeline.py`
- Classifies queries into `CONVERSATION`, `FACT_LOOKUP`, or `SUMMARY`
- Uses score-filtered similarity search plus reranking for factual lookup
- Uses MMR retrieval for summary-style questions
- Enforces route-specific prompt behavior and citation validation
- Returns structured output:

```python
{
  "answer": "...",
  "query_type": "FACT_LOOKUP|SUMMARY|CONVERSATION",
  "routing": {
    "query_type": "...",
    "confidence": 0.0,
    "reason": "...",
    "route_source": "LLM_CLASSIFIER|RULE_FALLBACK"
  },
  "sources": {
    1: ("file.pdf", page),
    2: ("file.pdf", page)
  }
}
```

### `src/cli_chat.py`
- Runs interactive terminal Q&A
- Displays selected route, classifier source, confidence, and reason
- Displays assistant answer
- Displays citations as `[id] file | page`
- Shows route-aware citation output for retrieval vs conversation mode

### `streamlit_app.py`
- Provides a browser UI for PDF upload and question entry
- Rebuilds the vector store from uploaded PDFs
- Shows a scrollable answer history with route metadata
- Displays citations and sources for each response

## Prerequisites

1. Install **Python 3.11**
2. Install [Ollama](https://ollama.com/)
3. Pull required models:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2:3b
ollama pull phi3:mini
```

## Setup

From project root:

```powershell
py -3.11 -m venv rag_env311
.\rag_env311\Scripts\activate
python -m pip install -r requirements.txt
```

## Model Configuration

Model choices are stage-wise configurable through environment variables.
If a variable is not set, the current default is used.

```powershell
$env:RAG_EMBEDDING_MODEL = "nomic-embed-text"

$env:RAG_CLASSIFIER_MODEL = "phi3:mini"
$env:RAG_CLASSIFIER_TEMPERATURE = "0"

$env:RAG_RERANKER_MODEL = "phi3:mini"
$env:RAG_RERANKER_TEMPERATURE = "0"

$env:RAG_ANSWER_MODEL = "llama3.2:3b"
$env:RAG_ANSWER_TEMPERATURE = "0"

$env:RAG_CONVERSATION_MODEL = "llama3.2:3b"
$env:RAG_CONVERSATION_TEMPERATURE = "0.2"
```

You can also place the same keys in a local `.env` file.

Important:
- If you change `RAG_EMBEDDING_MODEL`, rebuild the FAISS index with `python src\retriever.py`
- For low-latency experiments, try smaller models at `RAG_CLASSIFIER_MODEL`, `RAG_RERANKER_MODEL`, or `RAG_ANSWER_MODEL` independently

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

### 3) Start Streamlit UI

```powershell
python -m streamlit run streamlit_app.py
```

## Typical Workflow

1. Add or update PDFs in `data/`
2. Re-index with `python src\retriever.py`
3. Query with `python src\cli_chat.py`

## Notes

- Rebuild vector store whenever source PDFs change
- Rebuild vector store whenever `RAG_EMBEDDING_MODEL` changes
- Keep Ollama running while indexing/chatting
- Tune chunk size or score threshold to balance recall and precision
- Default query classification model: `phi3:mini`
- Default fact reranking model: `phi3:mini`
- Default answer and conversation model: `llama3.2:3b`
- Default embedding model: `nomic-embed-text`
- Fact lookup uses reranking after initial similarity retrieval
- Summary queries use MMR retrieval with diversified chunk selection


## Learnings
See [RAG Challenges & Solutions](docs/RAG_Challenges_and_Solutions.md)
