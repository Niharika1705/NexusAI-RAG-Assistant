import os
import sys
import logging
import streamlit as st
import uuid
import shutil

# Ensure project root is in sys.path so backend imports work cleanly when running streamlit from anywhere
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.ingest.pipeline import run_ingestion, save_uploaded_file
from backend.rag.generator import query_rag
from backend.rag.vectorstore import get_vectorstore

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="NexusAI | Premium RAG Chatbot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Stunning Dark Mode, Glassmorphism, Gradients, Outfit & JetBrains Mono Fonts)
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Base typography and background */
html, body, [class*="css"] {
    font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #e2e8f0;
    background-color: #0b0f19;
}

/* App container */
.stApp {
    background: radial-gradient(circle at 15% 20%, rgba(56, 189, 248, 0.08) 0%, rgba(11, 15, 25, 1) 45%),
                radial-gradient(circle at 85% 80%, rgba(168, 85, 247, 0.08) 0%, rgba(11, 15, 25, 1) 50%);
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.75) !important;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

section[data-testid="stSidebar"] .stMarkdown h1, 
section[data-testid="stSidebar"] .stMarkdown h2, 
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #f8fafc;
    font-weight: 600;
}

/* Header styling */
.header-container {
    padding: 1.5rem 0 2rem 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    margin-bottom: 2rem;
}
.app-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.app-subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    margin-top: 0.3rem;
}

/* Chat message cards */
[data-testid="stChatMessage"] {
    background: rgba(30, 41, 59, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 16px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    transition: transform 0.2s ease, border-color 0.2s ease;
}
[data-testid="stChatMessage"]:hover {
    border-color: rgba(56, 189, 248, 0.3);
}

/* Source citations badge container */
.citation-box {
    margin-top: 1rem;
    padding: 0.85rem 1rem;
    background: rgba(15, 23, 42, 0.65);
    border-left: 3px solid #38bdf8;
    border-radius: 8px;
    font-size: 0.9rem;
}
.citation-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #38bdf8;
    margin-bottom: 0.5rem;
    font-weight: 600;
}
.citation-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.15), rgba(129, 140, 248, 0.15));
    border: 1px solid rgba(56, 189, 248, 0.3);
    color: #e0f2fe;
    padding: 0.3rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-family: 'JetBrains Mono', monospace;
    margin: 0.25rem 0.4rem 0.25rem 0;
    transition: background 0.2s ease;
}
.citation-badge:hover {
    background: rgba(56, 189, 248, 0.25);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.2rem !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.35) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px 0 rgba(79, 70, 229, 0.5) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(15, 23, 42, 0.4);
    border: 1px dashed rgba(56, 189, 248, 0.3);
    border-radius: 12px;
    padding: 1rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Sidebar: Knowledge Base Manager
with st.sidebar:
    st.markdown("### ⚡ Nexus RAG Control")
    st.caption("Production-Ready Knowledge Base & Indexing Manager")
    st.divider()

    kb_dir = os.path.join(PROJECT_ROOT, "backend", "knowledge_base", st.session_state.session_id)
    if not os.path.exists(kb_dir):
        os.makedirs(kb_dir, exist_ok=True)

    pdf_files = [f for f in os.listdir(kb_dir) if f.lower().endswith(".pdf")]
    
    st.markdown("#### 📂 Indexed Knowledge Base")
    if pdf_files:
        for file_name in pdf_files:
            file_path = os.path.join(kb_dir, file_name)
            size_kb = os.path.getsize(file_path) / 1024
            st.markdown(f"📄 **{file_name}** `({size_kb:.1f} KB)`")
    else:
        st.info("No PDF files found in `./knowledge_base/`.")

    st.divider()
    st.markdown("#### 📤 Upload & Re-Index")
    uploaded_files = st.file_uploader(
        "Upload new PDF documents:",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload files directly to knowledge_base and re-index Chroma Cloud."
    )

    if uploaded_files:
        new_files = [uf for uf in uploaded_files if uf.name not in st.session_state.processed_files]
        if new_files:
            with st.status("Re-indexing Knowledge Base...", expanded=True) as status:
                st.write("Checking uploaded files...")
                for uf in new_files:
                    save_uploaded_file(uf, kb_dir=kb_dir)
                    st.write(f"Saved `{uf.name}`.")
                    st.session_state.processed_files.add(uf.name)
                
                st.write("Running PDF splitting and embedding pipeline...")
                try:
                    count = run_ingestion(kb_dir=kb_dir, reset=True, session_id=st.session_state.session_id)
                    status.update(label=f"Successfully indexed {count} chunks!", state="complete", expanded=False)
                    st.success(f"Indexed {count} chunks across {len(os.listdir(kb_dir))} files.")
                except Exception as e:
                    status.update(label="Indexing failed!", state="error", expanded=True)
                    st.error(f"Error during ingestion: {str(e)}")

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        # Clean up session data
        try:
            vs = get_vectorstore()
            vs._collection.delete(where={"session_id": st.session_state.session_id})
        except Exception as e:
            logger.warning(f"Could not clear vectorstore for session {st.session_state.session_id}: {e}")
            
        shutil.rmtree(kb_dir, ignore_errors=True)
        
        st.session_state.messages = []
        st.session_state.processed_files = set()
        st.rerun()

    st.divider()
    st.markdown("---")
    st.caption("Powered by **LangChain**, **Chroma Cloud**, **Sentence Transformers**, & **OpenRouter LLM**.")

# Main Header
st.markdown("""
<div class="header-container">
    <div class="app-title">NexusAI RAG Assistant</div>
    <div class="app-subtitle">Ask questions with context-bounded precision and instant verified document citations.</div>
</div>
""", unsafe_allow_html=True)

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            sources_html = "".join(
                f'<span class="citation-badge">📄 {src["source"]} (Page {src["page"]})</span>'
                for src in msg["sources"]
            )
            st.markdown(f"""
            <div class="citation-box">
                <div class="citation-title">Verified Citations</div>
                <div>{sources_html}</div>
            </div>
            """, unsafe_allow_html=True)

# Chat Input & Processing
if query := st.chat_input("Ask a question about your indexed PDF documents..."):
    # Append user prompt to history and display
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant chunks and generating answer..."):
            try:
                result = query_rag(question=query, k=5, session_id=st.session_state.session_id)
                answer = result.get("answer", "No answer generated.")
                sources = result.get("sources", [])
            except Exception as e:
                logger.error(f"Error answering query: {str(e)}", exc_info=True)
                answer = f"⚠️ Error generating response: {str(e)}"
                sources = []

        st.markdown(answer)
        if sources:
            sources_html = "".join(
                f'<span class="citation-badge">📄 {src["source"]} (Page {src["page"]})</span>'
                for src in sources
            )
            st.markdown(f"""
            <div class="citation-box">
                <div class="citation-title">Verified Citations</div>
                <div>{sources_html}</div>
            </div>
            """, unsafe_allow_html=True)

    # Save assistant response to session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
