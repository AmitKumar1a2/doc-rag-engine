import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings

load_dotenv()


@dataclass(frozen=True)
class ChatModelConfig:
    model: str
    temperature: float


def _get_env_str(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    return value or default


def _get_env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default

    try:
        return float(raw_value)
    except ValueError:
        return default


def get_embedding_model_name() -> str:
    return _get_env_str("RAG_EMBEDDING_MODEL", "nomic-embed-text")


def get_classifier_model_config() -> ChatModelConfig:
    return ChatModelConfig(
        model=_get_env_str("RAG_CLASSIFIER_MODEL", "phi3:mini"),
        temperature=_get_env_float("RAG_CLASSIFIER_TEMPERATURE", 0.0),
    )


def get_reranker_model_config() -> ChatModelConfig:
    return ChatModelConfig(
        model=_get_env_str("RAG_RERANKER_MODEL", "phi3:mini"),
        temperature=_get_env_float("RAG_RERANKER_TEMPERATURE", 0.0),
    )


def get_answer_model_config() -> ChatModelConfig:
    return ChatModelConfig(
        model=_get_env_str("RAG_ANSWER_MODEL", "llama3.2:3b"),
        temperature=_get_env_float("RAG_ANSWER_TEMPERATURE", 0.0),
    )


def get_conversation_model_config() -> ChatModelConfig:
    return ChatModelConfig(
        model=_get_env_str("RAG_CONVERSATION_MODEL", "llama3.2:3b"),
        temperature=_get_env_float("RAG_CONVERSATION_TEMPERATURE", 0.2),
    )


def create_chat_llm(config: ChatModelConfig) -> ChatOllama:
    return ChatOllama(model=config.model, temperature=config.temperature)


@lru_cache(maxsize=1)
def get_classifier_llm() -> ChatOllama:
    return create_chat_llm(get_classifier_model_config())


@lru_cache(maxsize=1)
def get_reranker_llm() -> ChatOllama:
    return create_chat_llm(get_reranker_model_config())


@lru_cache(maxsize=1)
def get_answer_llm() -> ChatOllama:
    return create_chat_llm(get_answer_model_config())


@lru_cache(maxsize=1)
def get_conversation_llm() -> ChatOllama:
    return create_chat_llm(get_conversation_model_config())


@lru_cache(maxsize=1)
def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=get_embedding_model_name())
