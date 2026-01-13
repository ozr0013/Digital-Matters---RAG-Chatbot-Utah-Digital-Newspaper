import os
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY"] = "false"
import time
import numpy as np
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

# ------------------------------------------
# CONFIG
# ------------------------------------------
CHUNK_DIR = r"E:\UDN_Project\chunked"
EMB_DIR   = r"E:\UDN_Project\embeddings"
CHROMA_PATH = r"E:\UDN_Project\chroma_db"
COLLECTION_NAME = "udn_archive"

# ------------------------------------------
# SETUP
# ------------------------------------------
from chromadb.config import Settings

client = chromadb.Client(
    Settings(
        persist_directory=CHROMA_PATH,
        anonymized_telemetry=False
    )
)

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn
)

def add_batch_safe(ids, embeddings, docs, metas):
    """Add small batches to avoid memory issues"""
    batch_size = 2000
    for i in range(0, len(ids), batch_size):
        j = i + batch_size
        collection.add(
            ids=ids[i:j],
            embeddings=embeddings[i:j],
            documents=docs[i:j],
            metadatas=metas[i:j],
        )

# ------------------------------------------
# MIGRATION
# ------------------------------------------
files = sorted([f for f in os.listdir(EMB_DIR) if f.endswith(".npy")])
start_time = time.time()
total_added = 0

for idx, npy_file in enumerate(files, 1):
    base = npy_file.replace(".npy", "")
    csv_file = f"{base}.csv"
    csv_path = os.path.join(CHUNK_DIR, csv_file)
    npy_path = os.path.join(EMB_DIR, npy_file)

    if not os.path.exists(csv_path):
        print(f"Missing CSV for {npy_file}, skipping.")
        continue

    try:
        X = np.load(npy_path)
        df = pd.read_csv(csv_path, dtype=str)
        texts = df.iloc[:, -1].astype(str).tolist()
        pdf_links = df.iloc[:, 8].fillna("").astype(str).tolist()
        ids = [f"{base}_{i}" for i in range(len(texts))]
        metas = [{"pdf": pdf_links[i]} for i in range(len(pdf_links))]

        # check if first id already exists → skip file
        try:
            existing = collection.get(ids=[ids[0]])
            if existing and existing.get("ids"):
                print(f" {npy_file} already in DB, skipping.")
                continue
        except Exception:
            pass

        t0 = time.time()
        add_batch_safe(ids, X.tolist(), texts, metas)
        t1 = time.time()

        added_now = len(texts)
        total_added += added_now
        elapsed = (t1 - start_time) / 60
        eta = elapsed / idx * (len(files) - idx)
        print(f" {npy_file}: added {added_now:,} chunks "
              f"({idx}/{len(files)}) | total {total_added:,} | "
              f"{elapsed:.1f} min elapsed, ~{eta:.1f} min left")
    except Exception as e:
        print(f"Error on {npy_file}: {e}")

print(f"\n Migration complete! Total {total_added:,} chunks added.")
print(f"DB saved at: {CHROMA_PATH}")
