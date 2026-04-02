# Utah Digital Newspapers RAG Chatbot — Full Onboarding

## What Is This Project?

This is a **RAG (Retrieval-Augmented Generation) chatbot** built on top of the **Utah Digital Newspapers archive** — a massive collection of historical newspaper scans from across Utah, going back over 100 years. The OCR text from those scans is searchable, but raw search is bad at understanding meaning. This chatbot lets you ask natural language questions like *"What happened during the 1918 flu epidemic in Salt Lake City?"* and get a real, cited answer grounded in actual newspaper articles.

**The core idea:** instead of asking an LLM to "know" history from training data, we retrieve the actual historical newspaper text first, then ask the LLM to read it and answer. The LLM never makes things up — it only synthesizes from what it retrieved.

---

## The Two Phases

### Phase 1 — Offline Data Pipeline (run once, takes hours)
You collect the data, process it, and build the search index. This only runs when you're setting up or updating the dataset.

### Phase 2 — Runtime (the chatbot)
The Flask web app loads the pre-built index and answers questions in real time.

---

## File-by-File Breakdown

### Root Level

**`app.py`**
The entry point. Starts the Flask web server, initializes the chatbot and LLM, and defines the API routes:
- `GET /` — serves the chat UI
- `POST /api/chat` — takes a question, returns an answer + sources
- `GET /api/health` — health check
- `GET /api/stats` — how many documents are loaded

**`requirements.txt`**
Python dependencies. Install with `pip install -r requirements.txt`.

**`.env`** *(not committed, you create this)*
Where you put your `GROQ_API_KEY`. The app reads it on startup.

---

### `src/` — All the logic

#### Data Collection

**`src/fetch_udn.py`**
Hits the Utah Digital Newspapers public API (`api.lib.utah.edu/udn/v1`) and downloads raw article data as JSON and CSV. This is how you get the raw newspaper data in the first place. You typically only run this once.

#### Data Preparation

**`src/chunking.py`**
Takes the big raw CSV (millions of articles) and splits every article's OCR text into 500-character chunks with 50-character overlap. Why chunks? Because embeddings work best on short pieces of text, and LLMs have context limits. Outputs hundreds of smaller `udn_chunks_partN.csv` files into `E:\UDN_Project\chunked\`.

**`src/embeddings.py`**
Earlier/simpler version of the embedding builder. Takes chunk CSVs and encodes each chunk into a 384-dimension vector using `all-MiniLM-L6-v2`. Saves results as `.npy` files.

**`src/build_embeddings_full.py`**
The production-scale version of embeddings.py. Handles GPU acceleration, auto-adjusts batch size if you run out of VRAM, skips already-processed files so you can resume if it crashes, and saves checkpoints every 25 files. This is the one you run overnight to build the full index.

#### The Vector Store

**`src/vectorstore_faiss.py`**
The heart of the retrieval system. Loads all `.npy` embedding files into a FAISS index, stores metadata (title, date, paper, article ID) in SQLite, and looks up full chunk text from the CSV on demand. Has two modes:
- Small dataset → `IndexFlatIP` (exact search)
- Full dataset → `IVF+PQ` (compressed, handles 300M+ vectors)

**`src/vectorstore.py`**
Legacy version that used ChromaDB instead of FAISS. Still here from the previous migration. Not used by the main app anymore.

#### The Chatbot

**`src/rag_chatbot_faiss.py`**
The `RAGChatbot` class. Ties everything together:
1. Takes a user question
2. Embeds it with SentenceTransformer
3. Searches the FAISS vectorstore for top-5 matching chunks
4. Formats results with titles, dates, relevance scores, and links back to the original article on `newspapers.lib.utah.edu`

**`src/rag_chatbot.py`**
Legacy version that used ChromaDB. Not used by the main app.

**`src/ollama_processor.py`**
The LLM layer. Takes the raw retrieval results and sends them to an LLM with a prompt like: *"Here are 5 newspaper excerpts. Read them carefully and answer the user's question. Cite your sources."* Supports:
- **Groq** (default) — free cloud API, fast, uses `llama-3.3-70b-versatile`
- **Ollama** — local LLM fallback, uses `llama3.2` running on your machine

#### Legacy / Migration

**`src/migrate_to_chroma.py`**
The script that loaded data into ChromaDB when the project used that backend. Not needed now but kept for reference.

**`src/inspect_chroma.py`**
Debug tool for inspecting what's in the old ChromaDB database.

#### Inspection / Debug Tools

**`src/inspect_faiss_index.py`**
Prints stats about the FAISS index — how many vectors, what dimensions, etc. Useful to verify the index built correctly.

**`src/inspect_data.py`**
Prints stats about the raw CSV data — row counts, column names, sample rows.

**`src/chunk_Checker.py`**
Sanity-checks the chunked CSV files to verify they look correct before running expensive embedding.

**`src/config.py`**
Shared path configuration (data dirs, etc.).

#### Test Scripts

**`src/rag_faiss_test.py`**
End-to-end test of the FAISS RAG pipeline without the web server. Run this to verify retrieval is working.

**`src/build_embeddings_test.py`**
Small-scale test of the embedding + FAISS build process on a subset of files.

**`src/ollama_test.py`**
Tests the Ollama local LLM connection independently of the rest of the app.

---

### `templates/` and `static/` — The Frontend

**`templates/index.html`**
The chat UI. A single-page HTML file served by Flask. Has the input box, message history, and source cards.

**`static/script.js`**
Handles sending messages to `/api/chat`, receiving the JSON response, and rendering the answer + source cards in the UI.

**`static/style.css`**
Styling for the chat interface.

---

### `data/` — Sample Data

**`data/udn_docs_sample.csv`** and **`data/udn_docs_sample.json`**
A small sample of 100 documents fetched via `fetch_udn.py`. Used for dev/testing without needing the full dataset.

---

### Other

**`launch_chatbot.bat`**
Windows batch file to start the app with one double-click.

**`create_shortcut.ps1`**
PowerShell script to create a desktop shortcut for the app.

**`venv/`**
The Python virtual environment. Never commit this — it's local only.

---

## The Data Flow in One Picture

```
Utah Digital Newspapers API
        ↓ fetch_udn.py
   udn.csv (raw data)
        ↓ chunking.py
   udn_chunks_partN.csv  (500-char text chunks with metadata)
        ↓ build_embeddings_full.py
   embeddings/*.npy  +  faiss_index/udn.index + udn.db
        ↓ (app starts)
   vectorstore_faiss.py loads index into memory
        ↓ (user asks question)
   rag_chatbot_faiss.py → embed query → FAISS search → top 5 chunks
        ↓
   ollama_processor.py → Groq LLM reads chunks → writes answer
        ↓
   Flask → JSON → Browser
```

---

## To Get Running

1. Set your Groq API key: create a `.env` file with `GROQ_API_KEY=your_key_here`
2. Install deps: `pip install -r requirements.txt`
3. Make sure `E:\UDN_Project\faiss_index\udn.index` exists (the pre-built index)
4. Run: `python app.py`
5. Open `http://localhost:5000`
