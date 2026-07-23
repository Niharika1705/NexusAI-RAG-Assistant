import os
import sys

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.rag.generator import query_rag

print("\n===== NexusAI OpenRouter RAG Chatbot CLI Started =====")
print("Type 'exit' or 'quit' to end conversation.\n")

while True:
    try:
        question = input("\nAsk a question: ").strip()
        if not question:
            continue
        if question.lower() in ["exit", "quit"]:
            print("Exiting chatbot. Goodbye!")
            break

        print("\nRetrieving context and querying OpenRouter LLM...")
        result = query_rag(question=question, k=5)
        
        print("\n=== Answer ===")
        print(result["answer"])
        
        if result.get("sources"):
            print("\n=== Sources ===")
            for idx, src in enumerate(result["sources"], 1):
                print(f"[{idx}] Document: {src['source']} (Page {src['page']})")
        else:
            print("\n(No specific sources cited or out of context)")
        print("=" * 40)
    except KeyboardInterrupt:
        print("\nExiting chatbot. Goodbye!")
        break
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {str(e)}")