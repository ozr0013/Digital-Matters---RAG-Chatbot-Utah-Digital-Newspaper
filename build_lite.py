"""
Build a self-contained lite dataset for cloud deployment.
Samples 25K docs across multiple files, stores everything in FAISS + SQLite.

Usage: python -u build_lite.py
Output: data/lite.index + data/lite.db
"""

import os
import sys
import numpy as np
import pandas as pd
import faiss
import sqlite3
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configuration
EMB_DIR = r"E:\UDN_Project\embeddings"
CHUNK_DIR = r"E:\UDN_Project\chunked"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
TARGET_DOCS = 25000
SAMPLE_FILES = 10  # Sample from this many files for variety

os.makedirs(OUTPUT_DIR, exist_ok=True)

INDEX_PATH = os.path.join(OUTPUT_DIR, "lite.index")
DB_PATH = os.path.join(OUTPUT_DIR, "lite.db")

# Remove old files
for f in [INDEX_PATH, DB_PATH]:
    if os.path.exists(f):
        os.remove(f)

print("=" * 50)
print("LITE INDEX BUILDER")
print("=" * 50)
print(f"Target: {TARGET_DOCS:,} docs from {SAMPLE_FILES} files")
print(f"Output: {OUTPUT_DIR}")
print("=" * 50)

# Pick files spread across the dataset
all_files = sorted([f for f in os.listdir(EMB_DIR) if f.endswith(".npy")])
total = len(all_files)
step = max(1, total // SAMPLE_FILES)
selected_files = [all_files[i] for i in range(0, total, step)][:SAMPLE_FILES]
docs_per_file = TARGET_DOCS // len(selected_files)

print(f"\nFound {total} total files, sampling {len(selected_files)} files")
print(f"~{docs_per_file} docs per file\n")

# Setup SQLite with text included
conn = sqlite3.connect(DB_PATH)
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("""
    CREATE TABLE documents (
        idx INTEGER PRIMARY KEY,
        article_id TEXT,
        article_title TEXT,
        date TEXT,
        paper TEXT,
        chunk_text TEXT
    )
""")

all_embeddings = []
global_idx = 0

for file_idx, npy_file in enumerate(selected_files):
    base = npy_file.replace(".npy", "")
    csv_path = os.path.join(CHUNK_DIR, f"{base}.csv")
    npy_path = os.path.join(EMB_DIR, npy_file)

    if not os.path.exists(csv_path):
        print(f"  Skip {npy_file}: no CSV")
        continue

    try:
        embeddings = np.load(npy_path).astype(np.float32)
        df = pd.read_csv(csv_path, dtype=str)

        if df.empty or len(df) != embeddings.shape[0]:
            print(f"  Skip {npy_file}: mismatch")
            continue

        # Sample rows
        n_rows = min(docs_per_file, len(df))
        indices = sorted(random.sample(range(len(df)), n_rows))

        sampled_emb = embeddings[indices]
        all_embeddings.append(sampled_emb)

        batch = []
        for i in indices:
            row = df.iloc[i]
            text = str(row.get("chunk_text", "") or "")[:1000]
            batch.append((
                global_idx,
                str(row.get("id", "") or ""),
                str(row.get("article_title", "") or ""),
                str(row.get("date", "") or ""),
                str(row.get("paper", "") or ""),
                text
            ))
            global_idx += 1

        conn.executemany("INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?)", batch)
        conn.commit()
        print(f"  [{file_idx + 1}/{len(selected_files)}] {npy_file}: {n_rows} docs sampled")

    except Exception as e:
        print(f"  Error {npy_file}: {e}")

conn.close()

if not all_embeddings:
    print("ERROR: No embeddings loaded!")
    sys.exit(1)

# Build FAISS flat index
print(f"\nBuilding FAISS index with {global_idx:,} docs...")
all_embeddings = np.vstack(all_embeddings)
faiss.normalize_L2(all_embeddings)

index = faiss.IndexFlatIP(384)
index.add(all_embeddings)
faiss.write_index(index, INDEX_PATH)

# Report sizes
index_size = os.path.getsize(INDEX_PATH) / 1024 / 1024
db_size = os.path.getsize(DB_PATH) / 1024 / 1024

print(f"\n{'=' * 50}")
print(f"LITE BUILD COMPLETE!")
print(f"{'=' * 50}")
print(f"Documents: {global_idx:,}")
print(f"Index: {INDEX_PATH} ({index_size:.1f} MB)")
print(f"Database: {DB_PATH} ({db_size:.1f} MB)")
print(f"Total: {index_size + db_size:.1f} MB")
print(f"\nReady for deployment!")
