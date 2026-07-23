import os
import logging
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

logger = logging.getLogger(__name__)

DEFAULT_KB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "knowledge_base"))

def load_all_pdfs(kb_dir: str = DEFAULT_KB_DIR, session_id: str = None) -> List[Document]:
    """
    Reads all PDF files from the specified knowledge base directory.
    Uses PyPDFLoader and attaches clean source names and 1-indexed page numbers.
    """
    if not os.path.exists(kb_dir):
        logger.warning(f"Knowledge base directory '{kb_dir}' does not exist. Creating it.")
        os.makedirs(kb_dir, exist_ok=True)
        return []

    pdf_files = [
        os.path.join(kb_dir, f) for f in os.listdir(kb_dir)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        logger.info(f"No PDF files found in '{kb_dir}'.")
        return []

    all_documents = []
    logger.info(f"Found {len(pdf_files)} PDF(s) in '{kb_dir}'. Starting loading...")

    for pdf_path in pdf_files:
        try:
            logger.info(f"Loading PDF: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            
            # Enrich metadata with clean file name and 1-indexed page number
            for doc in docs:
                source_path = doc.metadata.get("source", pdf_path)
                clean_name = os.path.basename(source_path)
                raw_page = doc.metadata.get("page", 0)
                page_num = raw_page + 1 if isinstance(raw_page, int) else raw_page
                
                doc.metadata["source"] = clean_name
                doc.metadata["page"] = page_num
                if session_id:
                    doc.metadata["session_id"] = session_id
                
                
            all_documents.extend(docs)
            logger.info(f"Successfully loaded {len(docs)} pages from '{os.path.basename(pdf_path)}'.")
        except Exception as e:
            logger.error(f"Error loading PDF '{pdf_path}': {str(e)}", exc_info=True)

    logger.info(f"Total pages loaded across all PDFs: {len(all_documents)}")
    return all_documents
