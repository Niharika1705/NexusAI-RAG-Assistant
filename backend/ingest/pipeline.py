import os
import logging
from typing import Any
from .pdf_loader import load_all_pdfs
from .splitter import split_documents
from backend.rag.vectorstore import store_documents

logger = logging.getLogger(__name__)

DEFAULT_KB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "knowledge_base"))

def save_uploaded_file(uploaded_file: Any, kb_dir: str = DEFAULT_KB_DIR) -> str:
    """
    Saves an uploaded file (e.g., from Streamlit uploader) into the knowledge_base directory.
    Returns the absolute destination path.
    """
    if not os.path.exists(kb_dir):
        os.makedirs(kb_dir, exist_ok=True)

    file_path = os.path.join(kb_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    logger.info(f"Saved uploaded file '{uploaded_file.name}' to '{file_path}'.")
    return file_path

def run_ingestion(
    kb_dir: str = DEFAULT_KB_DIR,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    reset: bool = True
) -> int:
    """
    Runs the full ingestion pipeline:
    1. Loads all PDFs from kb_dir.
    2. Splits them into chunks using RecursiveCharacterTextSplitter.
    3. Generates embeddings and stores them in Chroma Cloud.
    Returns the total number of indexed chunks.
    """
    logger.info("Starting ingestion pipeline...")
    
    # Step 1: Read all PDFs
    documents = load_all_pdfs(kb_dir=kb_dir)
    if not documents:
        logger.warning("No documents loaded. Ingestion pipeline terminated early.")
        return 0

    # Step 2: Split documents
    chunks = split_documents(
        documents=documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    if not chunks:
        logger.warning("No chunks created after splitting. Ingestion pipeline terminated early.")
        return 0

    # Step 3: Store in Chroma
    count = store_documents(chunks=chunks, reset=reset)
    logger.info(f"Ingestion pipeline completed successfully! Total chunks indexed: {count}")
    return count
