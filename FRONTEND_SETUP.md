# 🚀 Running the Frontend

This guide explains how to set up and run the web-based chatbot frontend for the Digital Matters RAG Chatbot project.

## Prerequisites

Ensure you have Python 3.8+ installed and all dependencies from `requirements.txt` are installed:

```bash
pip install -r requirements.txt
```

You'll need to install Flask and Flask-CORS if not already installed:

```bash
pip install flask flask-cors
```

## Project Structure

```
Digital-Matters---RAG-Chatbot-Utah-Digital-Newspaper/
├── app.py                    # Flask web server (main entry point)
├── requirements.txt          # Python dependencies
├── static/                   # Frontend static files
│   ├── style.css            # Chatbot styling
│   ├── script.js            # Frontend JavaScript logic
│   └── ...
├── templates/               # HTML templates
│   └── index.html           # Chatbot interface
└── src/                     # Backend RAG chatbot code
    ├── rag_chatbot.py       # Main chatbot logic
    ├── vectorstore.py       # Vector database integration
    └── ...
```

## Setup Steps

### 1. Install Dependencies

```bash
cd "c:\Users\u1529771\Desktop\ORAG\Digital-Matters---RAG-Chatbot-Utah-Digital-Newspaper"
pip install -r requirements.txt
pip install flask flask-cors
```

### 2. Configure Your RAG Chatbot Backend

Open `app.py` and uncomment the RAG chatbot import (around line 11):

```python
# Uncomment these lines and adjust based on your actual RAG chatbot implementation
# from rag_chatbot import RAGChatbot
# chatbot = RAGChatbot()
```

Update the `get_chatbot_response()` function to use your actual RAG chatbot:

```python
def get_chatbot_response(user_query):
    """Get a response from the RAG chatbot"""
    # Replace with your actual implementation
    response = chatbot.query(user_query)
    return {
        "answer": response["answer"],
        "sources": response["sources"]
    }
```

### 3. Run the Application

```bash
python app.py
```

The server will start on `http://localhost:5000`

## Accessing the Chatbot

1. Open your web browser
2. Navigate to `http://localhost:5000`
3. Start asking questions about the Utah Digital Newspapers!

## Features

✨ **Modern Chat Interface**
- Clean, responsive design
- Dark mode support
- Mobile-friendly layout
- Real-time message updates

📱 **Responsive Design**
- Works on desktop, tablet, and mobile
- Sidebar navigation on mobile
- Touch-friendly buttons

💬 **Chat Functionality**
- Send and receive messages
- Display source citations
- Chat history saved to browser localStorage
- Example queries for new users

🔍 **RAG Integration**
- Semantic search through vector database
- Document retrieval and ranking
- Source attribution

## Customization

### Changing the Port

In `app.py`, modify the port in the last line:

```python
app.run(debug=True, host='0.0.0.0', port=8000)  # Change 5000 to your desired port
```

### Styling

Edit `static/style.css` to customize:
- Colors (CSS variables at the top)
- Fonts and sizes
- Layout and spacing
- Animations

### HTML Structure

Edit `templates/index.html` to:
- Add more example queries
- Change the title or subtitle
- Modify sidebar content
- Update welcome message

## API Endpoints

### POST `/api/chat`

Send a question to the chatbot.

**Request:**
```json
{
    "message": "What articles mention women's suffrage in Utah?"
}
```

**Response:**
```json
{
    "answer": "The chatbot's response with cited information...",
    "sources": [
        {
            "title": "Article Title",
            "snippet": "Relevant excerpt from the article..."
        }
    ]
}
```

### GET `/api/health`

Health check endpoint.

**Response:**
```json
{
    "status": "healthy",
    "service": "Digital Matters RAG Chatbot"
}
```

## Troubleshooting

### Port Already in Use

If port 5000 is already in use, either:
1. Change the port in `app.py`
2. Find and stop the process using the port:
   ```bash
   # Windows PowerShell
   Get-Process | Where-Object {$_.port -eq 5000}
   ```

### CORS Issues

If you encounter CORS errors, ensure `flask-cors` is installed and `CORS(app)` is called in `app.py`.

### CSS/JS Not Loading

Clear your browser cache (Ctrl+Shift+Delete) and reload the page.

### Backend Not Responding

Ensure your RAG chatbot is properly integrated in the `get_chatbot_response()` function and the necessary data files are in place.

## Production Deployment

For production use:

1. Set `debug=False` in `app.py`
2. Use a production WSGI server like Gunicorn:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

3. Consider using a reverse proxy (Nginx) for better performance

4. Enable HTTPS with SSL certificates

5. Add authentication if needed

## Support

For issues or questions, refer to the main README.md or contact Digital Matters at the University of Utah.

---

**Built with ❤️ for Digital Matters**  
University of Utah
