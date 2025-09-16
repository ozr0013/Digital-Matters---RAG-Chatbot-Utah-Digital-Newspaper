# 📰 Digital-Matters RAG Chatbot — Utah Digital Newspapers

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Build](https://img.shields.io/badge/Build-Passing-brightgreen)](#)

**Author:** Omar Rizwan  
**Manager:** Rebekah Cummings
**Organization:** Digital Matters, University of Utah  
**Project Type:** Retrieval-Augmented Generation (RAG) Chatbot  
**Goal:** Help students, researchers, and the public explore the Utah Digital Newspapers archive through a conversational AI assistant.

---

## 📖 Overview

The **Digital-Matters RAG Chatbot** is an AI-powered assistant that allows users to search and explore the **Utah Digital Newspapers** collection using natural language questions.

Instead of manually browsing thousands of articles, users can ask:

> “Show me articles about women’s suffrage in Utah from the 1920s.”  
> “Find newspapers mentioning the 1918 Spanish flu in Salt Lake City.”

The chatbot uses a **Retrieval-Augmented Generation (RAG)** pipeline to:
- Embed and index historical newspaper articles.
- Retrieve the most relevant documents using semantic search.
- Generate concise, cited answers grounded in real data.

---

## 🧠 Architecture

       ┌──────────────────┐
       │  User Question   │
       └─────────┬────────┘
                 │
          (Convert to embedding)
                 │
       ┌─────────▼──────────┐
       │   Vector Database  │  ← FAISS / Weaviate
       └─────────┬──────────┘
                 │
         (Top-k Retrieved Docs)
                 │
       ┌─────────▼──────────┐
       │  Cross-Encoder     │  ← reranks results
       └─────────┬──────────┘
                 │
    (Pass top documents to LLM)
                 │
       ┌─────────▼──────────┐
       │     Chatbot LLM     │  ← Llama-3.1-8B-Instruct (LoRA tuned)
       └─────────┬──────────┘
                 │
         (Answer with citations)
                 │
       ┌─────────▼──────────┐
       │    Web Frontend     │  ← Flask / Streamlit
       └────────────────────┘

**Main components:**
- **Embeddings:** `intfloat/e5-base` (fine-tuned for newspaper content)
- **Vector Store:** FAISS (local) or Weaviate (cloud)
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Chat LLM:** `Llama-3.1-8B-Instruct` (LoRA fine-tuned for style & citations)
- **Frameworks:** LangChain / LlamaIndex
- **Interface:** Flask or Streamlit chatbot UI

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/OMAR-RIZWAN/Digital-Matters--RAG-Chatbot-Utah-Digital-Newspaper.git
cd Digital-Matters--RAG-Chatbot-Utah-Digital-Newspaper

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
