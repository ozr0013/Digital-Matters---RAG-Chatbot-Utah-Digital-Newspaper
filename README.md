# Digital-Matters RAG Chatbot - Utah Digital Newspapers

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Author:** Omar Rizwan
**Manager:** Rebekah Cummings
**Organization:** Digital Matters, University of Utah
**Project Type:** Retrieval-Augmented Generation (RAG) Chatbot
**Goal:** Help students, researchers, and the public explore the Utah Digital Newspapers archive through a conversational AI assistant.

---

## Overview

The **Digital-Matters RAG Chatbot** is an AI-powered assistant that allows users to search and explore the **Utah Digital Newspapers** collection using natural language questions.

Instead of manually browsing thousands of articles, users can ask:

> "Show me articles about women's suffrage in Utah from the 1920s."
> "Find newspapers mentioning the 1918 Spanish flu in Salt Lake City."

The chatbot retrieves the most relevant newspaper excerpts using semantic search, then uses an LLM to synthesize an intelligent, cited answer from the sources.

---

## Architecture

```
       +------------------+
       |  User Question   |
       +--------+---------+
                |
         (Sentence Transformer)
                |
       +--------v---------+
       |   FAISS Index     |  <-- IVF+PQ compressed, handles 300M+ vectors
       +--------+---------+
                |
        (Top-k Retrieved Docs)
                |
       +--------v---------+
       |   Groq LLM API   |  <-- llama-3.3-70b-versatile (free cloud)
       +--------+---------+
                |
        (Synthesized answer with citations)
                |
       +--------v---------+
       |   Flask Frontend  |  <-- Web chatbot UI
       +------------------+
```

**Main components:**
- **Embeddings:** `all-MiniLM-L6-v2` (384-dim, via sentence-transformers)
- **Vector Store:** FAISS with IVF+PQ compression + SQLite metadata
- **LLM:** Groq API (`llama-3.3-70b-versatile`) or Ollama (local fallback)
- **Interface:** Flask web application with chat UI
- **Data:** ~2,999 chunked CSV files with ~300M newspaper article chunks

---

## Installation

```bash
# Clone the repository
git clone https://github.com/ozr0013/Digital-Matters---RAG-Chatbot-Utah-Digital-Newspaper.git
cd Digital-Matters---RAG-Chatbot-Utah-Digital-Newspaper

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # On Mac/Linux: source venv/bin/activate

# Install dependencies
pip install flask flask-cors sentence-transformers faiss-cpu numpy pandas groq
```

---

## Setup

### 1. Data Preparation

The project expects pre-processed data in the following structure:

```
E:\UDN_Project\
  embeddings\    # .npy files (pre-computed embeddings)
  chunked\       # .csv files (chunked article text + metadata)
```

Each CSV contains columns: `id`, `article_title`, `date`, `paper`, `chunk_text`.

### 2. Groq API Key (Free)

Sign up at [console.groq.com](https://console.groq.com) and get a free API key.

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_key_here
```

### 3. Build the FAISS Index

For quick testing (50 files, ~5M docs):
```bash
python app.py
```
The app auto-builds a quick-start index on first run.

For the full dataset (all 2,999 files, ~300M docs):
```bash
python -u build_index.py
```
This takes ~2-3 hours and creates a compressed IVF+PQ index.

### 4. Run the App

```bash
python app.py
```

Open http://localhost:5000 in your browser.

---

## Project Structure

```
app.py                      # Flask web server (main entry point)
build_index.py              # Full FAISS index builder (run once)
src/
  vectorstore_faiss.py      # FAISS vector store with IVF+PQ + SQLite
  rag_chatbot_faiss.py      # RAG chatbot with semantic search
  ollama_processor.py       # LLM processor (Groq cloud / Ollama local)
  chunking.py               # Text chunking utilities
static/
  script.js                 # Frontend chat logic
  style.css                 # Styling
templates/
  index.html                # Chat interface
test_chatbot.py             # Integration tests
.env                        # API keys (not committed)
```

---

## How It Works

1. **User asks a question** via the web chat interface
2. **Sentence Transformer** encodes the query into a 384-dim embedding
3. **FAISS** finds the top-k most similar newspaper chunks (~300M vectors indexed)
4. **SQLite** looks up metadata (title, date, paper) for matched chunks
5. **CSV lookup** retrieves full text on-demand (keeps memory usage low)
6. **Groq LLM** reads the sources and synthesizes an intelligent answer
7. **Frontend** displays the answer with source citations and links to originals

Each search result includes a direct link to the original article on [newspapers.lib.utah.edu](https://newspapers.lib.utah.edu).

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Chat interface |
| `/api/chat` | POST | Send a query, get answer + sources |
| `/api/health` | GET | System health check |
| `/api/stats` | GET | Index statistics |

### Example request:
```json
POST /api/chat
{"message": "What articles mention women's suffrage in Utah?"}
```

---

## Scaling

The system is designed to handle the full UDN archive (~300M chunks):

- **IVF+PQ compression:** Reduces index from ~450GB (flat) to ~15GB
- **SQLite metadata:** Lightweight storage (~2GB for all metadata)
- **On-demand text lookup:** Full text loaded from CSV only during search
- **Quick-start mode:** 50-file subset for development/testing
