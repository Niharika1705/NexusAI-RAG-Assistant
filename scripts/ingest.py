import os
import sys
import logging

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.ingest.pipeline import run_ingestion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)

def main():
    print("===== NexusAI Knowledge Base Ingestion Script =====")
    kb_dir = os.path.join(PROJECT_ROOT, "backend", "knowledge_base")
    
    count = run_ingestion(kb_dir=kb_dir, reset=True)
    if count > 0:
        print(f"\n[SUCCESS] Successfully indexed {count} chunks into Chroma.")
    else:
        print("\n[WARNING] No chunks indexed. Check your knowledge base directory or PDFs.")

if __name__ == "__main__":
    main()