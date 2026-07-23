import logging
from typing import List, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma
from chromadb import CloudClient
from backend.config import (
    COLLECTION_NAME,
    CHROMA_API_KEY,
    CHROMA_TENANT,
    CHROMA_DATABASE,
)
from .embeddings import get_embedding_model

logger = logging.getLogger(__name__)

def get_chroma_client():
    """
    Returns a Chroma client.
    Connects to Chroma Cloud using CHROMA_API_KEY, CHROMA_TENANT, and CHROMA_DATABASE.
    """
    if not (CHROMA_API_KEY and CHROMA_TENANT and CHROMA_DATABASE):
        raise ValueError("Chroma Cloud credentials missing in .env file (CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE).")

    logger.info(f"Connecting to Chroma Cloud (Tenant: {CHROMA_TENANT}, Database: {CHROMA_DATABASE})...")
    try:
        client = CloudClient(
            tenant=CHROMA_TENANT,
            database=CHROMA_DATABASE,
            api_key=CHROMA_API_KEY
        )
        return client
    except Exception as e:
        logger.error(f"Error connecting to Chroma Cloud: {str(e)}", exc_info=True)
        raise

def get_vectorstore(collection_name: str = COLLECTION_NAME) -> Chroma:
    """
    Returns the initialized Chroma vectorstore with the shared embedding model.
    """
    embedding_model = get_embedding_model()
    client = get_chroma_client()
    
    vectorstore = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embedding_model
    )
    return vectorstore

def store_documents(chunks: List[Document], collection_name: str = COLLECTION_NAME, reset: bool = True) -> int:
    """
    Stores document chunks into Chroma Cloud or DB.
    If reset is True, clears existing collection first before storing new chunks.
    Returns the number of chunks stored.
    """
    if not chunks:
        logger.warning("No chunks provided to store.")
        return 0

    logger.info(f"Storing {len(chunks)} chunks into Chroma collection '{collection_name}' (reset={reset})...")
    client = get_chroma_client()
    embedding_model = get_embedding_model()

    if reset:
        try:
            client.delete_collection(name=collection_name)
            logger.info(f"Existing collection '{collection_name}' deleted for re-indexing.")
        except Exception:
            logger.info(f"Collection '{collection_name}' did not exist or could not be deleted before reset.")

    vectorstore = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embedding_model
    )

    batch_size = 100
    total_stored = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        try:
            vectorstore.add_documents(batch)
            total_stored += len(batch)
            logger.info(f"Stored batch {i // batch_size + 1}: {len(batch)} chunks added.")
        except Exception as e:
            logger.error(f"Error storing batch {i // batch_size + 1}: {str(e)}", exc_info=True)
            raise

    logger.info(f"Successfully indexed and stored {total_stored} chunks across collection '{collection_name}'.")
    return total_stored

def get_retriever(k: int = 5, collection_name: str = COLLECTION_NAME):
    """
    Returns a LangChain retriever configured to fetch the top `k` (default: 5) most relevant chunks.
    """
    vectorstore = get_vectorstore(collection_name=collection_name)
    return vectorstore.as_retriever(search_kwargs={"k": k})
