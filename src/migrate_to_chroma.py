import os
import sys
import time
import numpy as np
import pandas as pd
import chromadb

# ------------------------------------------
# CONFIGURATION
# ------------------------------------------
CHUNK_DIR = r"D:\UDN_Project\chunked"
EMB_DIR   = r"D:\UDN_Project\embeddings"
DB_PATH   = r"D:\UDN_Project\chroma_db"
COLLECTION_NAME = "udn_archive"

BATCH_SIZE = 5400          # Max safe batch size for Chroma
PROGRESS_FILE = "completed_files.txt"

# ------------------------------------------
# SETUP CHROMA DB
# ------------------------------------------
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(COLLECTION_NAME)
print(f"\nChroma is saving to: {DB_PATH}")

# ------------------------------------------
# PROGRESS TRACKING
# ------------------------------------------
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE, "r") as f:
        completed = set(line.strip() for line in f if line.strip())
else:
    completed = set()

# ------------------------------------------
# FILE DISCOVERY
# ------------------------------------------
embedding_files = sorted([f for f in os.listdir(EMB_DIR) if f.endswith(".npy")])
print(f"\nResuming migration — {len(completed)} files already completed.")
print(f"Found {len(embedding_files)} embedding files total.\n")

# ------------------------------------------
# MIGRATION LOOP (SINGLE-THREADED)
# ------------------------------------------
start_time = time.time()

for i, file_name in enumerate(embedding_files, start=1):
    if file_name in completed:
        print(f"[{i}/{len(embedding_files)}] Skipped: {file_name}")
        continue

    try:
        idx = int(file_name.replace("udn_chunks_part", "").replace(".npy", ""))
        chunk_path = os.path.join(CHUNK_DIR, f"udn_chunks_part{idx}.csv")
        emb_path = os.path.join(EMB_DIR, file_name)

        if not os.path.exists(chunk_path):
            print(f"[{i}/{len(embedding_files)}] Missing chunk file: {chunk_path}")
            continue

        embeddings = np.load(emb_path, mmap_mode="r")
        df = pd.read_csv(chunk_path)

        # Detect text column automatically
        text_col_candidates = ["chunk_text", "text", "content", "body"]
        text_col = next((c for c in text_col_candidates if c in df.columns), None)
        if not text_col:
            print(f"[{i}/{len(embedding_files)}] Error: No text column found in {file_name}")
            continue

        texts = df[text_col].astype(str).tolist()
        metadatas = df.to_dict(orient="records")
        ids = [f"{idx}_{j}" for j in range(len(df))]

        # Insert in safe batches
        for start in range(0, len(df), BATCH_SIZE):
            end = start + BATCH_SIZE
            collection.add(
                ids=ids[start:end],
                embeddings=embeddings[start:end].tolist(),
                metadatas=metadatas[start:end],
                documents=texts[start:end]
            )

        # Record as completed
        with open(PROGRESS_FILE, "a") as f:
            f.write(file_name + "\n")

        print(f"[{i}/{len(embedding_files)}] Done: {file_name}")

    except Exception as e:
        print(f"[{i}/{len(embedding_files)}] Error processing {file_name}: {e}")

elapsed = time.time() - start_time
print(f"\nMigration complete in {elapsed/60:.2f} minutes.")
