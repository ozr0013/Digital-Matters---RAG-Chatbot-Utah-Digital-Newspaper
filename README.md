# Digital-Matters RAG Chatbot - Utah Digital Newspapers

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Author:** Omar Rizwan
**Manager:** Rebekah Cummings
**Organization:** Digital Matters, University of Utah
**Project Type:** Retrieval-Augmented Generation (RAG) Chatbot
**Goal:** Help students, researchers, and the public explore the Utah Digital Newspapers archive.

---

## Overview

The **Digital-Matters RAG Chatbot** is an AI-powered assistant that allows users to search
and explore the **Utah Digital Newspapers** collection using natural language questions.

Instead of manually browsing thousands of articles, users can ask:

> "Show me articles about women's suffrage in Utah from the 1920s."
> "Find newspapers mentioning the 1918 Spanish flu in Salt Lake City."

The chatbot retrieves the most relevant newspaper excerpts using semantic search,
then uses an LLM to synthesize an intelligent, cited answer from the sources.

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
       |   FAISS Index     |  <-- IVF+PQ (large) or FlatIP (small)
       +--------+---------+
                |
        (Top-k Retrieved Docs)
                |
       +--------v---------+
       |   Groq LLM API   |  <-- llama-3.3-70b-versatile (free cloud)
       +--------+---------+       or Ollama (local fallback)
                |
        (Synthesized answer with citations)
                |
       +--------v---------+
       |   Flask Frontend  |  <-- Web chatbot UI
       +------------------+
```

**Main components:**
- **Embeddings:** `all-MiniLM-L6-v2` (384-dim, via sentence-transformers)
- **Vector Store:** FAISS with IVF+PQ (large dataset) or IndexFlatIP (small dataset) + SQLite metadata
- **LLM:** Groq API (`llama-3.3-70b-versatile`) or Ollama (local fallback)
- **Interface:** Flask web application with chat UI
- **Data:** ~2,999 chunked CSV files with ~300M newspaper article chunks

---

## Installation

```bash
# Clone the repository
git clone https://github.com/ozr0913/Digital-Matters---RAG-Chatbot-Utah-Digital-Newspaper.git
cd Digital-Matters---RAG-Chatbot-Utah-Digital-Newspaper

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # On Mac/Linux: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Setup

### 1. Configure Data Paths

Edit `src/config.py` to point to your data directories:

```python
CHUNK_DIR = r"E:\UDN_Project\chunked"       # directory of chunked CSV files
EMB_DIR   = r"E:\UDN_Project\embeddings"    # directory of .npy embedding files
CHROMA_PATH = r"E:\UDN_Project\chroma_db"  # ChromaDB storage (legacy, optional)
```

The project expects pre-processed data in the following structure:

```
E:\UDN_Project\
  embeddings\    # .npy files (pre-computed embeddings)
  chunked\       # .csv chunk files (article text + metadata)
  faiss_index\   # saved FAISS index + SQLite DB (auto-generated)
```

Each CSV contains columns: `id`, `article_title`, `date`, `paper`, `chunk_text`.

### 2. Groq API Key (Free)

Sign up at [console.groq.com](https://console.groq.com) and get a free API key.

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_key_here
```

### 3. Build the FAISS Index

For the full dataset (all ~2,999 files), run the index builder from the project root:
```bash
python build_index.py
```
This loads all `.npy` embedding files, builds a compressed IVF+PQ FAISS index, and saves it with SQLite metadata. Uses GPU if available (CUDA), falls back to CPU. **Expect several hours for the full dataset** — run overnight.

Alternatively, to also regenerate embedding files from raw CSVs:
```bash
python src/build_embeddings_full.py
```

If no saved index is found, `app.py` will auto-build a quick-start index from existing `.npy` files on first run.

### 4. Run the App

```bash
python app.py
```

Or on Windows, double-click `launch_chatbot.bat`.

Open http://localhost:5000 in your browser.

> For new contributors, see [ONBOARDING.md](ONBOARDING.md) for a full walkthrough.

---

## Project Structure

```
app.py                          # Flask web server (main entry point)
build_index.py                  # Full FAISS index builder (run once, overnight)
test_chatbot.py                 # End-to-end tests for the chatbot
launch_chatbot.bat              # Windows convenience launcher
create_shortcut.ps1             # PowerShell script to create desktop shortcut
src/
  config.py                     # Centralized config (data paths, model settings)
  vectorstore_faiss.py          # FAISS vector store with IVF+PQ + SQLite metadata
  rag_chatbot_faiss.py          # RAG chatbot with semantic search (active)
  ollama_processor.py           # LLM processor (Groq cloud / Ollama local)
  build_embeddings_full.py      # Embedding generator + FAISS index builder
  build_embeddings_test.py      # Test embedding builder (small sample)
  chunking.py                   # Text chunking pipeline for raw articles
  chunk_Checker.py              # Utility: validate chunk files
  embeddings.py                 # Embedding utilities
  rag_chatbot.py                # RAG chatbot (ChromaDB version, legacy)
  vectorstore.py                # ChromaDB vector store (legacy)
  migrate_to_chroma.py          # One-time migration tool (ChromaDB, legacy)
  rag_faiss_test.py             # Integration tests
  inspect_faiss_index.py        # Utility: inspect FAISS index stats
  inspect_chroma.py             # Utility: inspect ChromaDB (legacy)
  inspect_data.py               # Utility: inspect raw data
  fetch_udn.py                  # Utility: fetch data from UDN API
  ollama_test.py                # Utility: test Ollama connection
data/
  udn_docs_sample.csv           # Small sample dataset for testing
  udn_docs_sample.json
static/
  script.js                     # Frontend chat logic
  style.css                     # Styling
templates/
  index.html                    # Chat interface
.env                            # API keys (not committed)
ONBOARDING.md                   # New contributor guide
FRONTEND_SETUP.md               # Frontend setup instructions
```

---

## How It Works

1. **User asks a question** via the web chat interface
2. **Sentence Transformer** encodes the query into a 384-dim embedding
3. **FAISS** finds the top-k most similar newspaper chunks
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

- **IVF+PQ compression:** Reduces index from ~450GB (flat) to ~15GB for large datasets (>200 files)
- **IndexFlatIP:** Used for small datasets (<200 files) for exact search
- **SQLite metadata:** Lightweight storage (~2GB for all metadata)
- **On-demand text lookup:** Full text loaded from CSV only during search
