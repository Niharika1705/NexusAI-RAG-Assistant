import logging
from typing import List
from langchain_core.documents import Document
from .vectorstore import get_retriever, get_vectorstore

logger = logging.getLogger(__name__)

def retrieve_relevant_chunks(query: str, k: int = 5, session_id: str = None) -> List[Document]:
    """
    Retrieves the top `k` (default: 5) most relevant document chunks from Chroma Cloud
    for a given query string (Requirement 6). Filters by session_id if provided.
    """
    logger.info(f"Retrieving top {k} relevant chunks for query: '{query}' (session_id={session_id})")
    try:
        retriever = get_retriever(k=k, session_id=session_id)
        docs = retriever.invoke(query)
        logger.info(f"Successfully retrieved {len(docs)} chunks.")
        for idx, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            logger.debug(f"Chunk {idx} [Source: {source}, Page: {page}]: {doc.page_content[:100]}...")
        return docs
    except Exception as e:
        logger.error(f"Error during retrieval: {str(e)}", exc_info=True)
        return []
