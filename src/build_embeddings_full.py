from sentence_transformers import SentenceTransformer
import numpy as np, pandas as pd, faiss, os, time

# -------------------------------------------
# CONFIG
# -------------------------------------------
base_dir = r"D:\UDN_Project\chunked"        # folder with all chunk CSVs
save_dir = r"D:\UDN_Project\embeddings"     # folder to store .npy embeddings
index_path_partial = r"D:\UDN_Project\udn_faiss_partial.index"
index_path_full = r"D:\UDN_Project\udn_faiss_full.index"
AUTOSAVE_INTERVAL = 25                      # save every 25 files
# -------------------------------------------

os.makedirs(save_dir, exist_ok=True)

print("🔹 Loading embedding model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
dim = 384
index = faiss.IndexFlatL2(dim)

files = sorted([f for f in os.listdir(base_dir) if f.endswith(".csv")])
print(f"Found {len(files)} chunk files\n")

batch_count = 0
total_vectors = 0
start_time = time.time()

for i, f in enumerate(files):
    path = os.path.join(base_dir, f)
    print(f"\n🔹 Processing {f} ({i+1}/{len(files)}) ...")
    df = pd.read_csv(path)

    texts = df['chunk_text'].astype(str).tolist()
    embeddings = model.encode(texts, show_progress_bar=True)

    # Save per-file embeddings (optional but useful)
    np.save(os.path.join(save_dir, f.replace('.csv', '.npy')), embeddings)

    # Add to FAISS index
    index.add(embeddings)
    total_vectors += len(embeddings)
    print(f"Added {len(embeddings):,} vectors from {f} (Total so far: {total_vectors:,})")

    batch_count += 1
    if batch_count % AUTOSAVE_INTERVAL == 0:
        faiss.write_index(index, index_path_partial)
        print(f" Autosaved partial index after {batch_count} files at {index_path_partial}")

elapsed = time.time() - start_time
faiss.write_index(index, index_path_full)

print(f"\n Finished! Final FAISS index saved → {index_path_full}")
print(f"Total vectors indexed: {total_vectors:,}")
print(f" Total time: {elapsed/3600:.2f} hours")
