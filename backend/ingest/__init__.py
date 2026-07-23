"""
Backend ingest module for RAG Chatbot.
Handles PDF loading, text splitting, and ingestion pipeline orchestration.
"""
from .pdf_loader import load_all_pdfs
from .splitter import split_documents
from .pipeline import run_ingestion, save_uploaded_file

__all__ = [
    "load_all_pdfs",
    "split_documents",
    "run_ingestion",
    "save_uploaded_file",
]
