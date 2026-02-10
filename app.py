"""
Flask web application for the Utah Digital Newspapers RAG Chatbot
Provides REST API endpoints and serves the frontend chatbot interface
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import sys
from pathlib import Path

# Load .env file if it exists
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, val = line.split('=', 1)
                os.environ.setdefault(key.strip(), val.strip())

# Add src directory to path so we can import the chatbot modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# ======================
# RAG CHATBOT SETUP
# ======================

# Configuration - Using FAISS (no migration needed!)
EMB_DIR = r"E:\UDN_Project\embeddings"
CHUNK_DIR = r"E:\UDN_Project\chunked"
INDEX_PATH = r"E:\UDN_Project\faiss_index\udn.index"

# If full index exists, load it. Otherwise use a separate quick-start index.
INDEX_EXISTS = os.path.exists(INDEX_PATH)
if not INDEX_EXISTS:
    INDEX_PATH = os.path.join(os.path.dirname(INDEX_PATH) or ".", "udn_quick.index")
    INDEX_EXISTS = os.path.exists(INDEX_PATH)
MAX_FILES = None if INDEX_EXISTS else 50

# Initialize chatbot
chatbot = None
chatbot_error = None

try:
    from rag_chatbot_faiss import RAGChatbot
    if INDEX_EXISTS:
        print(f"Loading saved index from {INDEX_PATH}...")
    else:
        print(f"No saved index. Building with {MAX_FILES} files for quick start...")
        print("TIP: Run 'python build_index.py' to build full index (run overnight)")
    chatbot = RAGChatbot(EMB_DIR, CHUNK_DIR, INDEX_PATH, max_files=MAX_FILES)
    print(f"RAG Chatbot ready with {chatbot.get_stats()['total_documents']:,} documents")
except Exception as e:
    chatbot_error = str(e)
    print(f"Warning: Could not initialize RAG chatbot: {e}")
    print("Check that EMB_DIR and CHUNK_DIR paths are correct.")

# ======================
# LLM INTEGRATION (Groq cloud or Ollama local)
# ======================
# Groq: Free cloud API - get key at https://console.groq.com
# Ollama: Local LLM - install from https://ollama.ai
LLM_BACKEND = os.environ.get("LLM_BACKEND", "groq")  # "groq" or "ollama"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

llm_processor = None
try:
    from ollama_processor import LLMProcessor
    llm_processor = LLMProcessor(
        backend=LLM_BACKEND,
        groq_api_key=GROQ_API_KEY,
    )
    if llm_processor.is_available():
        print(f"LLM available: {LLM_BACKEND} backend")
    else:
        if LLM_BACKEND == "groq":
            print("Groq not available. Set GROQ_API_KEY environment variable.")
            print("Get a free key at: https://console.groq.com")
        else:
            print("Ollama not running. Start with: ollama serve")
        llm_processor = None
except Exception as e:
    print(f"LLM integration disabled: {e}")
    llm_processor = None


def get_chatbot_response(user_query: str, use_llm: bool = True) -> dict:
    """
    Get a response from the RAG chatbot.
    Optionally processes through LLM (Groq/Ollama) for intelligent answers.
    """
    # If chatbot is ready, use it
    if chatbot is not None:
        try:
            # Get RAG results
            response = chatbot.query(user_query, top_k=5)

            # Process through LLM for intelligent answer
            if use_llm and llm_processor:
                try:
                    response = llm_processor.process_rag_response(
                        query=user_query,
                        rag_response=response
                    )
                except Exception as llm_err:
                    print(f"LLM processing failed: {llm_err}")
                    # Continue with raw response

            return response
        except Exception as e:
            return {
                "answer": f"An error occurred while searching: {str(e)}",
                "sources": []
            }

    # Mock response when chatbot is not ready
    return {
        "answer": (
            f"The search system is not yet initialized. "
            f"Please run the migration script to populate the database. "
            f"Error: {chatbot_error or 'Unknown'}"
        ),
        "sources": [
            {
                "title": "Setup Required",
                "snippet": "Run 'python src/migrate_to_chroma.py' to load the newspaper archive into the database.",
                "date": "",
                "paper": "System Message",
                "relevance": "N/A"
            }
        ]
    }


# ======================
# ROUTES
# ======================

@app.route('/')
def index():
    """Serve the main chatbot interface."""
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    API endpoint to handle chat messages.

    Request JSON:
    {
        "message": "user question here",
        "use_llm": true  (optional - enables Ollama processing)
    }

    Response JSON:
    {
        "answer": "chatbot response",
        "sources": [{"title": "...", "snippet": "...", "date": "...", "paper": "..."}],
        "llm_summary": "..." (if Ollama enabled)
    }
    """
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        use_llm = data.get('use_llm', True)  # Default to using LLM if available

        if not user_message:
            return jsonify({
                "error": "Message cannot be empty"
            }), 400

        # Get response from chatbot (with optional LLM processing)
        response = get_chatbot_response(user_message, use_llm=use_llm)

        return jsonify(response), 200

    except Exception as e:
        return jsonify({
            "error": f"An error occurred: {str(e)}"
        }), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    stats = chatbot.get_stats() if chatbot else {"status": "not_initialized", "error": chatbot_error}

    return jsonify({
        "status": "healthy" if chatbot else "degraded",
        "service": "Utah Digital Newspapers RAG Chatbot",
        "chatbot": stats,
        "llm": {
            "backend": LLM_BACKEND,
            "available": llm_processor is not None
        }
    }), 200


@app.route('/api/stats', methods=['GET'])
def stats():
    """Get statistics about the archive."""
    if chatbot:
        return jsonify(chatbot.get_stats()), 200
    else:
        return jsonify({
            "error": "Chatbot not initialized",
            "details": chatbot_error
        }), 503


# ======================
# MAIN
# ======================

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)

    # Run the Flask application
    app.run(debug=True, host='0.0.0.0', port=5000)
