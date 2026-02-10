"""
Flask web application for the Utah Digital Newspapers RAG Chatbot
Supports two modes:
  - Lite mode (cloud): Self-contained data/lite.index + data/lite.db
  - Full mode (local): FAISS index on E:\UDN_Project with CSV files
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import sys

# Load .env file if it exists
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, val = line.split('=', 1)
                os.environ.setdefault(key.strip(), val.strip())

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# ======================
# RAG CHATBOT SETUP
# ======================

# Auto-detect mode: lite (cloud) vs full (local)
BASE_DIR = os.path.dirname(__file__)
LITE_INDEX = os.path.join(BASE_DIR, "data", "lite.index")
LITE_MODE = os.path.exists(LITE_INDEX)

# Full mode paths
FULL_EMB_DIR = r"E:\UDN_Project\embeddings"
FULL_CHUNK_DIR = r"E:\UDN_Project\chunked"
FULL_INDEX_PATH = r"E:\UDN_Project\faiss_index\udn.index"

chatbot = None
vectorstore = None
model = None
chatbot_error = None

try:
    from sentence_transformers import SentenceTransformer

    if LITE_MODE:
        # --- LITE MODE (cloud deployment) ---
        from vectorstore_lite import LiteVectorStore
        print("LITE MODE: Loading self-contained index...")
        vectorstore = LiteVectorStore(LITE_INDEX)
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print(f"Lite chatbot ready: {vectorstore.count():,} documents")
    else:
        # --- FULL MODE (local with E:\ drive data) ---
        from rag_chatbot_faiss import RAGChatbot
        INDEX_PATH = FULL_INDEX_PATH
        INDEX_EXISTS = os.path.exists(INDEX_PATH)
        if not INDEX_EXISTS:
            INDEX_PATH = os.path.join(os.path.dirname(INDEX_PATH) or ".", "udn_quick.index")
            INDEX_EXISTS = os.path.exists(INDEX_PATH)
        MAX_FILES = None if INDEX_EXISTS else 50

        if INDEX_EXISTS:
            print(f"FULL MODE: Loading saved index from {INDEX_PATH}...")
        else:
            print(f"FULL MODE: Building with {MAX_FILES} files...")
        chatbot = RAGChatbot(FULL_EMB_DIR, FULL_CHUNK_DIR, INDEX_PATH, max_files=MAX_FILES)
        print(f"Full chatbot ready: {chatbot.get_stats()['total_documents']:,} documents")

except Exception as e:
    chatbot_error = str(e)
    print(f"Warning: Could not initialize chatbot: {e}")

# ======================
# LLM INTEGRATION
# ======================
LLM_BACKEND = os.environ.get("LLM_BACKEND", "groq")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

llm_processor = None
try:
    from ollama_processor import LLMProcessor
    llm_processor = LLMProcessor(backend=LLM_BACKEND, groq_api_key=GROQ_API_KEY)
    if llm_processor.is_available():
        print(f"LLM available: {LLM_BACKEND}")
    else:
        if LLM_BACKEND == "groq":
            print("Groq not available. Set GROQ_API_KEY env var.")
        llm_processor = None
except Exception as e:
    print(f"LLM disabled: {e}")
    llm_processor = None

# Utah Digital Newspapers URL
UDN_BASE_URL = "https://newspapers.lib.utah.edu/details?id="


def get_chatbot_response(user_query: str, use_llm: bool = True) -> dict:
    """Get a response - works in both lite and full mode."""

    if LITE_MODE and vectorstore and model:
        # Lite mode: manual query pipeline
        try:
            query_embedding = model.encode(user_query).tolist()
            results = vectorstore.search(query_embedding, n_results=5)

            if not results["ids"]:
                return {"answer": "No relevant articles found.", "sources": []}

            sources = []
            for doc, meta, dist in zip(results["documents"], results["metadatas"], results["distances"]):
                title = meta.get("article_title", "")
                if not title or title in ("nan", "None", ""):
                    title = "Untitled Article"
                date = meta.get("date", "")
                if date and "T" in date:
                    date = date.split("T")[0]
                article_id = meta.get("article_id", "")
                similarity = max(0, 1 - dist) * 100
                sources.append({
                    "title": title,
                    "snippet": doc[:300] + "..." if len(doc) > 300 else doc,
                    "date": date,
                    "paper": meta.get("paper", "Unknown"),
                    "article_id": article_id,
                    "link": f"{UDN_BASE_URL}{article_id}" if article_id else "",
                    "relevance": f"{similarity:.0f}%"
                })

            # Build default answer
            n = len(sources)
            papers = set(s["paper"] for s in sources if s["paper"])
            answer = f"Found {n} relevant article{'s' if n > 1 else ''} from the Utah Digital Newspapers archive."
            if papers:
                answer += f" Sources: {', '.join(sorted(papers)[:3])}."

            response = {"answer": answer, "sources": sources}

            if use_llm and llm_processor:
                try:
                    response = llm_processor.process_rag_response(user_query, response)
                except Exception as e:
                    print(f"LLM error: {e}")

            return response
        except Exception as e:
            return {"answer": f"Search error: {str(e)}", "sources": []}

    elif chatbot:
        # Full mode: use RAGChatbot
        try:
            response = chatbot.query(user_query, top_k=5)
            if use_llm and llm_processor:
                try:
                    response = llm_processor.process_rag_response(user_query, response)
                except Exception as e:
                    print(f"LLM error: {e}")
            return response
        except Exception as e:
            return {"answer": f"Search error: {str(e)}", "sources": []}

    return {
        "answer": f"Chatbot not initialized. Error: {chatbot_error or 'Unknown'}",
        "sources": []
    }


# ======================
# ROUTES
# ======================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        use_llm = data.get('use_llm', True)

        if not user_message:
            return jsonify({"error": "Message cannot be empty"}), 400

        response = get_chatbot_response(user_message, use_llm=use_llm)
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route('/api/health', methods=['GET'])
def health():
    doc_count = 0
    if LITE_MODE and vectorstore:
        doc_count = vectorstore.count()
    elif chatbot:
        doc_count = chatbot.get_stats()['total_documents']

    return jsonify({
        "status": "healthy" if (vectorstore or chatbot) else "degraded",
        "mode": "lite" if LITE_MODE else "full",
        "documents": doc_count,
        "llm": {"backend": LLM_BACKEND, "available": llm_processor is not None}
    }), 200


@app.route('/api/stats', methods=['GET'])
def stats():
    if LITE_MODE and vectorstore:
        return jsonify({
            "total_documents": vectorstore.count(),
            "model": "all-MiniLM-L6-v2",
            "backend": "FAISS (lite)",
            "mode": "lite"
        }), 200
    elif chatbot:
        return jsonify(chatbot.get_stats()), 200
    return jsonify({"error": "Not initialized", "details": chatbot_error}), 503


# ======================
# MAIN
# ======================

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    debug = not LITE_MODE  # No debug in production
    app.run(debug=debug, host='0.0.0.0', port=port)
