import os
import sys
import logging
from typing import List, Dict, Any

# Ensure project root is in sys.path so 'backend.*' imports work cleanly when running uvicorn from the backend directory
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_DIR)

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.config import PROJECT_ROOT, COLLECTION_NAME
from backend.rag.generator import query_rag
from backend.ingest.pipeline import save_uploaded_file, run_ingestion

logger = logging.getLogger("backend.api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="NexusAI RAG Backend API",
    description="Production-Ready RAG Chatbot API using LangChain, Chroma Cloud, Sentence Transformers & OpenRouter LLM.",
    version="1.0.0"
)

# Enable CORS for frontend applications (Streamlit, Render, Vercel, localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class QueryRequest(BaseModel):
    question: str = Field(..., description="The user question to ask the RAG chatbot.")
    k: int = Field(5, description="Number of top chunks to retrieve.")

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

class IngestResponse(BaseModel):
    status: str
    message: str
    chunks_indexed: int

@app.get("/", tags=["Health"])
@app.get("/api/health", tags=["Health"])
async def health_check():
    """Returns the operational status of the RAG backend API."""
    return {
        "status": "online",
        "service": "NexusAI RAG Backend API",
        "collection": COLLECTION_NAME
    }

@app.post("/api/query", response_model=QueryResponse, tags=["RAG Query"])
async def execute_query(request: QueryRequest):
    """
    Executes a RAG query:
    1. Retrieves top k relevant document chunks from Chroma Cloud.
    2. Sends context and question to OpenRouter LLM.
    3. Returns answer along with exact source document names and page numbers.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    try:
        logger.info(f"Received query: '{request.question}' (k={request.k})")
        result = query_rag(question=request.question, k=request.k)
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
    except Exception as e:
        logger.error(f"Error executing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error during RAG execution: {str(e)}")

@app.post("/api/ingest", response_model=IngestResponse, tags=["Knowledge Base Ingestion"])
async def upload_and_ingest_pdf(
    file: UploadFile = File(...),
    reset: bool = Query(True, description="Whether to clear existing vector collection before indexing")
):
    """
    Uploads a new PDF file to backend/knowledge_base/ and runs the ingestion pipeline to index chunks into Chroma Cloud.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported for ingestion.")
    
    try:
        logger.info(f"Processing uploaded PDF: '{file.filename}' (reset={reset})")
        # Save file locally
        file_path = save_uploaded_file(uploaded_file=file)
        
        # Run ingestion pipeline
        count = run_ingestion(reset=reset)
        
        return IngestResponse(
            status="success",
            message=f"Successfully uploaded '{file.filename}' and indexed {count} chunks into Chroma Cloud.",
            chunks_indexed=count
        )
    except Exception as e:
        logger.error(f"Error during PDF ingestion: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
