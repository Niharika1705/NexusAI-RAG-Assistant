import logging
from typing import Any
from backend.config import EMBEDDING_MODEL_NAME, HF_API_KEY

logger = logging.getLogger(__name__)

# Singleton instance to ensure exact same embedding model is used for indexing and querying
_embedding_model_instance = None

def get_embedding_model() -> Any:
    """
    Returns the Sentence Transformer embedding model instance.
    Uses HuggingFaceEndpointEmbeddings (cloud API).
    
    Ensures the exact same embedding model instance is used across both indexing and querying.
    """
    global _embedding_model_instance
    if _embedding_model_instance is not None:
        return _embedding_model_instance

    if not HF_API_KEY:
        raise ValueError("HF_API_KEY is not set. Please set it in your .env file to use the HuggingFace API.")

    logger.info(f"Using cloud-hosted Sentence Transformer embedding endpoint: {EMBEDDING_MODEL_NAME}")
    
    from langchain_core.embeddings import Embeddings
    import requests

    class SimpleHFEmbeddings(Embeddings):
        def __init__(self, token: str, model: str):
            self.token = token
            self.url = f"https://router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction"
        
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            headers = {"Authorization": f"Bearer {self.token}"}
            resp = requests.post(self.url, headers=headers, json={"inputs": texts})
            if resp.status_code != 200:
                raise ValueError(f"HF API Error: {resp.status_code} - {resp.text}")
            return resp.json()
            
        def embed_query(self, text: str) -> list[float]:
            return self.embed_documents([text])[0]

    _embedding_model_instance = SimpleHFEmbeddings(token=HF_API_KEY, model=EMBEDDING_MODEL_NAME)
    return _embedding_model_instance
