"""
Hugging Face Spaces Entrypoint (Python SDK)
Hugging Face automatically runs `uvicorn app:app --host 0.0.0.0 --port 7860` when deploying as a Python Space.
"""
from backend.api import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api:app", host="0.0.0.0", port=7860, reload=False)
