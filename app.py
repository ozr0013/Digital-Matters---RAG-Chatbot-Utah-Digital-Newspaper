"""
Flask web application for the Digital Matters RAG Chatbot
Provides REST API endpoints and serves the frontend chatbot interface
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import sys
from pathlib import Path

# Add src directory to path so we can import the chatbot modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# You can import your RAG chatbot modules here
# from rag_chatbot import RAGChatbot  # Uncomment when rag_chatbot.py is ready
# chatbot = RAGChatbot()

# Mock responses for demonstration - replace with actual RAG chatbot logic
def get_chatbot_response(user_query):
    """
    Get a response from the RAG chatbot
    Replace this with your actual RAG chatbot implementation
    """
    # TODO: Replace with actual RAG chatbot logic
    return {
        "answer": f"I received your query: '{user_query}'. This is a mock response. Please implement the RAG chatbot backend.",
        "sources": [
            {"title": "Sample Article", "snippet": "This is a sample source from your vector database."}
        ]
    }

@app.route('/')
def index():
    """Serve the main chatbot interface"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    API endpoint to handle chat messages
    
    Request JSON:
    {
        "message": "user question here"
    }
    
    Response JSON:
    {
        "answer": "chatbot response",
        "sources": [{"title": "...", "snippet": "..."}]
    }
    """
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({
                "error": "Message cannot be empty"
            }), 400
        
        # Get response from chatbot
        response = get_chatbot_response(user_message)
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({
            "error": f"An error occurred: {str(e)}"
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Digital Matters RAG Chatbot"
    }), 200

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    # Run the Flask application
    # Set debug=False for production
    app.run(debug=True, host='0.0.0.0', port=5000)
