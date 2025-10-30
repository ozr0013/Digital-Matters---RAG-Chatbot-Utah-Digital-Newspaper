# rag_faiss_test.py
from sentence_transformers import SentenceTransformer
import faiss, numpy as np, pandas as pd, os, ollama

# ---------------------------------
# CONFIG
# ---------------------------------
INDEX_PATH = r"D:\UDN_Project\udn_faiss_full.index"   # your FAISS index
CHUNK_DIR  = r"D:\UDN_Project\chunked"                # folder with chunked text CSVs
MODEL_NAME = "mistral"                                # or "phi3"
TOP_K = 5

# ---------------------------------
# LOAD MODELS
# ---------------------------------
print("🔹 Loading embedding model and FAISS index...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index(INDEX_PATH)
print(f" FAISS index loaded ({index.ntotal:,} vectors)\n")

# ---------------------------------
# LOAD SOME TEXTS (you can load fewer if memory is tight)
# ---------------------------------
def load_texts(base_dir, limit_files=10):
    texts = []
    files = sorted([f for f in os.listdir(base_dir) if f.endswith(".csv")])[:limit_files]
    for f in files:
        try:
            df = pd.read_csv(os.path.join(base_dir, f), usecols=["chunk_text"])
            texts.extend(df["chunk_text"].astype(str).tolist())
        except Exception:
            pass
    return texts

print("🔹 Loading small sample of chunk texts...")
chunk_texts = load_texts(CHUNK_DIR)
print(f"Loaded {len(chunk_texts):,} sample chunks\n")

# ---------------------------------
# QUERY FUNCTION
# ---------------------------------
def query_ollama_rag(question):
    print(f" Query: {question}")
    q_emb = embedder.encode([question], convert_to_numpy=True)
    D, I = index.search(q_emb, TOP_K)
    context = "\n\n".join([chunk_texts[i] for i in I[0] if i < len(chunk_texts)])
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer concisely using only the context."
    
    response = ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
    answer = response["message"]["content"]
    print("\n💬 Ollama Answer:\n", answer)
    return answer

# ---------------------------------
# RUN TEST QUERY
# ---------------------------------
if __name__ == "__main__":
    query_ollama_rag("When was the Salt Lake Tribune founded?")
