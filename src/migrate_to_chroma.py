"""
ChromaDB Migration Script
Migrates chunked CSV files and their embeddings (.npy) to ChromaDB

Usage: python migrate_to_chroma.py
"""

import os
import time
import numpy as np
import pandas as pd
import chromadb

# ------------------------------------------
# CONFIG - Update these paths as needed
# ------------------------------------------
CHUNK_DIR = r"E:\UDN_Project\chunked"
EMB_DIR = r"E:\UDN_Project\embeddings"
CHROMA_PATH = r"E:\UDN_Project\chroma_db"
COLLECTION_NAME = "udn_archive"
BATCH_SIZE = 2000  # ChromaDB batch size for inserts

# ------------------------------------------
# SETUP - Use PersistentClient (not deprecated Client)
# ------------------------------------------
print(f"Initializing ChromaDB at: {CHROMA_PATH}")
os.makedirs(CHROMA_PATH, exist_ok=True)

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}  # cosine similarity for sentence embeddings
)

print(f"Collection '{COLLECTION_NAME}' has {collection.count()} documents")


def add_batch_safe(ids, embeddings, docs, metas, batch_size=BATCH_SIZE):
    """Add documents to ChromaDB in batches to avoid memory issues."""
    for i in range(0, len(ids), batch_size):
        j = min(i + batch_size, len(ids))
        collection.add(
            ids=ids[i:j],
            embeddings=embeddings[i:j],
            documents=docs[i:j],
            metadatas=metas[i:j],
        )


def check_already_ingested(base_name):
    """Check if a file has already been ingested by checking for its first ID."""
    test_id = f"{base_name}_0"
    try:
        existing = collection.get(ids=[test_id])
        return existing and existing.get("ids") and len(existing["ids"]) > 0
    except Exception:
        return False


# ------------------------------------------
# MIGRATION
# ------------------------------------------
def run_migration():
    # Get list of embedding files
    files = sorted([f for f in os.listdir(EMB_DIR) if f.endswith(".npy")])
    print(f"Found {len(files)} embedding files to process")

    if not files:
        print("No .npy files found in", EMB_DIR)
        return

    start_time = time.time()
    total_added = 0
    processed = 0
    skipped = 0
    errors = 0

    for idx, npy_file in enumerate(files, 1):
        base = npy_file.replace(".npy", "")
        csv_file = f"{base}.csv"
        csv_path = os.path.join(CHUNK_DIR, csv_file)
        npy_path = os.path.join(EMB_DIR, npy_file)

        # Check if CSV exists
        if not os.path.exists(csv_path):
            print(f"[SKIP] Missing CSV for {npy_file}")
            skipped += 1
            continue

        # Check if already ingested
        if check_already_ingested(base):
            print(f"[SKIP] Already ingested: {npy_file}")
            skipped += 1
            continue

        try:
            # Load embeddings
            X = np.load(npy_path)
            if X.ndim != 2:
                print(f"[ERROR] Invalid embedding shape in {npy_file}: {X.shape}")
                errors += 1
                continue

            # Load CSV
            try:
                df = pd.read_csv(csv_path, dtype=str)
            except Exception as csv_err:
                print(f"[WARN] Retrying CSV with python engine: {csv_file}")
                df = pd.read_csv(csv_path, dtype=str, engine="python")

            if df.empty:
                print(f"[SKIP] Empty CSV: {csv_file}")
                skipped += 1
                continue

            # Validate chunk_text column exists
            if "chunk_text" not in df.columns:
                print(f"[ERROR] No 'chunk_text' column in {csv_file}")
                print(f"  Available columns: {list(df.columns)}")
                errors += 1
                continue

            # Extract data
            texts = df["chunk_text"].fillna("").astype(str).tolist()

            # Validate row counts match
            if len(texts) != X.shape[0]:
                print(f"[ERROR] Count mismatch in {npy_file}: embeddings={X.shape[0]}, texts={len(texts)}")
                errors += 1
                continue

            # Build IDs and metadata
            ids = [f"{base}_{i}" for i in range(len(texts))]
            metadatas = []
            for i, row in df.iterrows():
                metadatas.append({
                    "article_id": str(row.get("id", "") or ""),
                    "article_title": str(row.get("article_title", "") or ""),
                    "date": str(row.get("date", "") or ""),
                    "paper": str(row.get("paper", "") or ""),
                    "chunk_index": str(row.get("chunk_index", "0") or "0"),
                    "source_file": base
                })

            # Add to ChromaDB
            add_batch_safe(ids, X.tolist(), texts, metadatas)

            total_added += len(ids)
            processed += 1

            # Progress reporting
            elapsed = (time.time() - start_time) / 60
            rate = total_added / elapsed if elapsed > 0 else 0
            remaining_files = len(files) - idx
            eta = (elapsed / processed) * remaining_files if processed > 0 else 0

            print(
                f"[{idx}/{len(files)}] {npy_file}: "
                f"+{len(ids)} chunks | "
                f"Total: {total_added:,} | "
                f"Rate: {rate:,.0f}/min | "
                f"ETA: {eta:.1f} min"
            )

        except Exception as e:
            print(f"[ERROR] {npy_file}: {e}")
            errors += 1
            continue

    # Final summary
    elapsed_total = (time.time() - start_time) / 60
    print("\n" + "=" * 50)
    print("MIGRATION COMPLETE")
    print("=" * 50)
    print(f"Files processed: {processed}")
    print(f"Files skipped: {skipped}")
    print(f"Errors: {errors}")
    print(f"Total chunks added: {total_added:,}")
    print(f"Total time: {elapsed_total:.1f} minutes")
    print(f"Collection size: {collection.count():,} documents")
    print(f"ChromaDB location: {CHROMA_PATH}")


if __name__ == "__main__":
    run_migration()
