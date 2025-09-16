# 📰 Digital-Matters–RAG-Chatbot–Utah-Digital-Newspaper

**Author:** Omar Rizwan  
**Organization:** Digital Matters, University of Utah  
**Project Type:** Retrieval-Augmented Generation (RAG) Chatbot  
**Goal:** Help students, researchers, and the public explore the Utah Digital Newspapers archive using a conversational AI assistant.

---

## 📖 Overview

The **Digital-Matters RAG Chatbot** is an AI-powered assistant that allows users to search and explore the **Utah Digital Newspapers** collection using natural language queries.

Instead of manually browsing thousands of articles, users can ask conversational questions like:

> “Show me articles about women’s suffrage in Utah from the 1920s.”  
> “Find newspapers mentioning the 1918 Spanish flu in Salt Lake City.”

The chatbot uses a **RAG (Retrieval-Augmented Generation)** pipeline to:
1. Embed and index historical newspaper articles.
2. Retrieve the most relevant content using semantic search.
3. Generate concise, cited answers grounded in real data.

---

## 🧠 Architecture

- **Embeddings:** `intfloat/e5-base` (fine-tuned for newspaper domain)
- **Vector Store:** FAISS (local) or Weaviate (cloud)
- **Retriever:** Dense vector similarity search + Cross-Encoder reranker
- **Chat Model:** `Llama-3.1-8B-Instruct` (QLoRA fine-tuned for style & citations)
- **Framework:** LangChain or LlamaIndex
- **Frontend:** Flask / Streamlit web app interface

**Pipeline Flow:**

