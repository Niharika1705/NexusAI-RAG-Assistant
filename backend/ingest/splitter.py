import logging
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Splits loaded documents into smaller chunks using RecursiveCharacterTextSplitter.
    Preserves document metadata on every chunk.
    """
    if not documents:
        logger.info("No documents provided to split.")
        return []

    logger.info(
        f"Splitting {len(documents)} documents with chunk_size={chunk_size} "
        f"and chunk_overlap={chunk_overlap}..."
    )

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} document pages.")
        return chunks
    except Exception as e:
        logger.error(f"Error while splitting documents: {str(e)}", exc_info=True)
        raise
