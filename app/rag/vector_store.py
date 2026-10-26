from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import (
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    QDRANT_URL,
)


def get_embeddings():

    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )