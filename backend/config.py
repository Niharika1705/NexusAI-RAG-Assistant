import os
from dotenv import load_dotenv

# Load environment variables from backend/.env (or fallback to root .env)
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    load_dotenv(env_path, override=True)
else:
    load_dotenv(override=True)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Chroma Configuration
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "rag_collection")

# Chroma Cloud Credentials
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")

# Embedding Configuration
HF_API_KEY = os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
if HF_API_KEY:
    os.environ["HF_TOKEN"] = HF_API_KEY

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-mpnet-base-v2")

# OpenRouter Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/auto")
