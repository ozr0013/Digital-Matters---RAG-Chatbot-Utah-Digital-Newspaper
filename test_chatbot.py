"""
Quick test script for the RAG chatbot.
Run this after migration to verify everything works.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

CHROMA_PATH = r"E:\UDN_Project\chroma_db"

print("=" * 50)
print("RAG CHATBOT TEST")
print("=" * 50)

# Test 1: Check ChromaDB connection
print("\n[1] Testing ChromaDB connection...")
try:
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection("udn_archive")
    doc_count = collection.count()
    print(f"    OK - Connected! Collection has {doc_count:,} documents")
except Exception as e:
    print(f"    FAIL - {e}")
    sys.exit(1)

# Test 2: Check embedding model
print("\n[2] Testing embedding model...")
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    test_embedding = model.encode("test query")
    print(f"    OK - Model loaded! Embedding dim: {len(test_embedding)}")
except Exception as e:
    print(f"    FAIL - {e}")
    sys.exit(1)

# Test 3: Test a search query
print("\n[3] Testing search query...")
try:
    query = "Utah history"
    query_emb = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )

    print(f"    OK - Search for '{query}' returned {len(results['ids'][0])} results:")
    for i, (doc, meta, dist) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ), 1):
        title = meta.get('article_title', 'Untitled')[:50]
        paper = meta.get('paper', 'Unknown')
        similarity = (1 - dist) * 100
        print(f"      {i}. {title}... ({paper}) - {similarity:.0f}% match")
except Exception as e:
    print(f"    FAIL - {e}")
    sys.exit(1)

# Test 4: Test RAGChatbot class
print("\n[4] Testing RAGChatbot class...")
try:
    from rag_chatbot import RAGChatbot
    chatbot = RAGChatbot(CHROMA_PATH)
    result = chatbot.query("women's suffrage", top_k=2)
    print(f"    OK - Chatbot works! Answer: {result['answer'][:100]}...")
    print(f"      Sources: {len(result['sources'])} found")
except Exception as e:
    print(f"    FAIL - {e}")
    sys.exit(1)

print("\n" + "=" * 50)
print("ALL TESTS PASSED!")
print("=" * 50)
print("\nYou can now run: python app.py")
print("Then open http://localhost:5000 in your browser")
