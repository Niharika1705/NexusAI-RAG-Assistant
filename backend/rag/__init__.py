"""
Backend RAG module containing embedding factory, vector store connection, retrieval, and OpenRouter generation logic.
"""
from .embeddings import get_embedding_model
from .vectorstore import get_vectorstore, store_documents, get_retriever
from .retriever import retrieve_relevant_chunks
from .generator import generate_answer, query_rag

__all__ = [
    "get_embedding_model",
    "get_vectorstore",
    "store_documents",
    "get_retriever",
    "retrieve_relevant_chunks",
    "generate_answer",
    "query_rag",
]
