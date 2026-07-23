import logging
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from backend.config import MISTRAL_API_KEY, MISTRAL_MODEL
from .retriever import retrieve_relevant_chunks

logger = logging.getLogger(__name__)

RAG_PROMPT_TEMPLATE = """You are an accurate AI assistant designed to answer questions based ONLY on the provided document context.

Instructions & Strict Rules:
1. Use ONLY the information provided in the context below to answer the user's question.
2. If the answer is not found in the context or cannot be directly inferred from the context, you MUST respond with EXACTLY this sentence:
"The answer is not available in the provided documents."
3. Do not assume or extrapolate facts outside the context. Do not include external knowledge.
4. Be clear, concise, and direct.

Context:
{context}

Question:
{question}

Answer:"""

def get_mistral_llm() -> ChatOpenAI:
    """
    Configures and returns the Mistral LLM using ChatOpenAI.
    Enforces Requirement 9: temperature=0, max_tokens=400.
    """
    if not MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY is missing! Please add it to your Streamlit Cloud Secrets or local .env file.")

    logger.info(f"Initializing Mistral LLM (model: {MISTRAL_MODEL}, temp=0, max_tokens=400)")

    return ChatOpenAI(
        model=MISTRAL_MODEL,
        base_url="https://api.mistral.ai/v1",
        api_key=MISTRAL_API_KEY,
        temperature=0,
        max_tokens=400
    )

def generate_answer(question: str, context_docs: List[Document]) -> Dict[str, Any]:
    """
    Generates an answer from the retrieved chunks using Mistral LLM and strictly formatted RAG prompt.
    Returns the answer and formatted source metadata (source document name and page number).
    """
    if not context_docs:
        logger.info("No context documents retrieved. Returning out-of-context fallback message.")
        return {
            "answer": "The answer is not available in the provided documents.",
            "sources": [],
            "chunks": []
        }

    context_str = "\n\n".join(
        f"--- Document: {doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', 'Unknown')}) ---\n{doc.page_content}"
        for doc in context_docs
    )

    prompt = RAG_PROMPT_TEMPLATE.format(context=context_str, question=question)

    try:
        llm = get_mistral_llm()
        response = llm.invoke(prompt)
        answer_text = response.content.strip()

        # Deduplicate sources preserving order
        sources = []
        seen = set()
        for doc in context_docs:
            src = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            key = (src, page)
            if key not in seen:
                seen.add(key)
                sources.append({"source": src, "page": page})

        return {
            "answer": answer_text,
            "sources": sources,
            "chunks": context_docs
        }
    except Exception as e:
        logger.error(f"Error generating answer with Mistral: {str(e)}", exc_info=True)
        return {
            "answer": f"Error communicating with Mistral LLM: {str(e)}",
            "sources": [],
            "chunks": context_docs
        }

def query_rag(question: str, k: int = 5, session_id: str = None) -> Dict[str, Any]:
    """
    High-level wrapper function:
    1. Retrieves top k relevant chunks from Chroma Cloud.
    2. Generates context-bounded answer with Mistral.
    3. Returns structured dict with answer and source citations.
    """
    chunks = retrieve_relevant_chunks(query=question, k=k, session_id=session_id)
    return generate_answer(question=question, context_docs=chunks)
