from sentence_transformers import SentenceTransformer
import torch, re, os, numpy as np, pandas as pd, faiss, time, gc

# -------------------------------------------
# CONFIG
# -------------------------------------------
base_dir = r"D:\UDN_Project\chunked"        # folder with all chunk CSVs
save_dir = r"D:\UDN_Project\embeddings"     # folder to store .npy embeddings
index_path_partial = r"D:\UDN_Project\udn_faiss_partial.index"
index_path_full = r"D:\UDN_Project\udn_faiss_full.index"
AUTOSAVE_INTERVAL = 25                      # save every 25 files
BATCH_SIZE = 8                             # start batch size (auto adjusts if OOM)
# -------------------------------------------

os.makedirs(save_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔹 Using device: {device}")
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
dim = 384
index = faiss.IndexFlatL2(dim)

# ---- Helper to extract numeric index for sorting ----
def extract_num(filename):
    m = re.search(r'(\d+)', filename)
    return int(m.group(1)) if m else -1

# ---- Gather and sort CSVs numerically ----
files = sorted(
    [f for f in os.listdir(base_dir) if f.endswith(".csv")],
    key=extract_num
)

# ---- Skip already processed embeddings ----
processed_files = {
    f.replace('.npy', '.csv') for f in os.listdir(save_dir) if f.endswith(".npy")
}
files = [f for f in files if f not in processed_files]
print(f"Found {len(files)} files to process (skipping {len(processed_files)} already done)\n")

batch_count = 0
total_vectors = 0
start_time = time.time()

for i, f in enumerate(files):
    path = os.path.join(base_dir, f)
    print(f"\n🔹 Processing {f} ({i+1}/{len(files)}) ...")

    # ---- Load CSV safely ----
    try:
        df = pd.read_csv(path, encoding_errors='ignore')
        if 'chunk_text' not in df.columns:
            print(f"Skipping {f} — no 'chunk_text' column found.")
            continue
        texts = df['chunk_text'].astype(str).tolist()
    except Exception as e:
        print(f" Error reading {f}: {e}")
        continue

    # ---- Encode with CUDA and retry logic ----
    success = False
    local_batch = BATCH_SIZE
    for attempt in range(3):
        try:
            embeddings = model.encode(
                texts,
                batch_size=local_batch,
                show_progress_bar=True,
                convert_to_numpy=True,
                device='cuda'
            )
            success = True
            break
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f" OOM on {f} (attempt {attempt+1}/3) → reducing batch size to {max(4, local_batch//2)}")
                torch.cuda.empty_cache()
                gc.collect()
                time.sleep(5)
                local_batch = max(4, local_batch // 2)
            else:
                print(f" RuntimeError on {f}: {e}")
                torch.cuda.empty_cache()
                gc.collect()
                break

    if not success:
        print(f" Skipping {f} after repeated OOMs.")
        continue

    # ---- Save embeddings + update FAISS index ----
    try:
        np.save(os.path.join(save_dir, f.replace('.csv', '.npy')), embeddings)
        index.add(embeddings)
        total_vectors += len(embeddings)
        print(f" Added {len(embeddings):,} vectors from {f} (Total so far: {total_vectors:,})")
    except Exception as e:
        print(f" Error saving or adding embeddings from {f}: {e}")
        continue

    # ---- Autosave & cleanup ----
    batch_count += 1
    if batch_count % AUTOSAVE_INTERVAL == 0:
        faiss.write_index(index, index_path_partial)
        print(f" Autosaved partial index → {index_path_partial}")

    del embeddings
    torch.cuda.empty_cache()
    gc.collect()

# ---- Final save ----
faiss.write_index(index, index_path_full)
elapsed = time.time() - start_time
print(f"\n Finished! Final FAISS index → {index_path_full}")
print(f"Total vectors indexed: {total_vectors:,}")
print(f" Total time: {elapsed/3600:.2f} hours")
