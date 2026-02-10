"""
Build the full FAISS index from ALL embedding files.
Uses compressed IVF+PQ index + SQLite metadata to handle 300M+ vectors.

Usage: python -u build_index.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from vectorstore_faiss import FAISSVectorStore

# Configuration
EMB_DIR = r"E:\UDN_Project\embeddings"
CHUNK_DIR = r"E:\UDN_Project\chunked"
INDEX_PATH = r"E:\UDN_Project\faiss_index\udn.index"

# Delete old index if exists (force full rebuild)
for old_file in [INDEX_PATH, INDEX_PATH.replace(".index", ".db"),
                 INDEX_PATH.replace(".index", "_meta.pkl")]:
    if os.path.exists(old_file):
        os.remove(old_file)
        print(f"Removed old: {old_file}")

print("=" * 60)
print("FAISS FULL INDEX BUILDER")
print("=" * 60)
print(f"Embeddings: {EMB_DIR}")
print(f"Chunks:     {CHUNK_DIR}")
print(f"Index:      {INDEX_PATH}")
print(f"DB:         {INDEX_PATH.replace('.index', '.db')}")
print("=" * 60)

files = [f for f in os.listdir(EMB_DIR) if f.endswith(".npy")]
print(f"\nFound {len(files)} embedding files")
print("Building compressed IVF+PQ index...\n")

store = FAISSVectorStore(
    emb_dir=EMB_DIR,
    chunk_dir=CHUNK_DIR,
    index_path=INDEX_PATH,
    max_files=None  # ALL files
)

print("\n" + "=" * 60)
print("BUILD COMPLETE!")
print("=" * 60)
print(f"Total documents: {store.count():,}")
print(f"Index: {INDEX_PATH}")
print(f"Metadata: {INDEX_PATH.replace('.index', '.db')}")
print("\nRun 'python app.py' to start the chatbot!")
