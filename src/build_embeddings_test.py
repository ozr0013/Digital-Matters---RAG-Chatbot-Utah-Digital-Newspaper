from sentence_transformers import SentenceTransformer
import numpy as np, pandas as pd, faiss, os

# Paths
base_dir = r"D:\UDN_Project\chunked"
save_dir = r"D:\UDN_Project\embeddings"
os.makedirs(save_dir, exist_ok=True)

# Load model
print(" Loading embedding model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
dim = 384  # output vector size
index = faiss.IndexFlatL2(dim)

# Collect files
files = sorted([f for f in os.listdir(base_dir) if f.endswith(".csv")])
print(f"Found {len(files)} chunk files")

# Process first 2 files for testing
for f in files[:2]:
    path = os.path.join(base_dir, f)
    print(f"🔹 Processing {f} ...")
    df = pd.read_csv(path)

    texts = df['chunk_text'].astype(str).tolist()
    embeddings = model.encode(texts, show_progress_bar=True)

    # Save embeddings separately
    np.save(os.path.join(save_dir, f.replace('.csv', '.npy')), embeddings)

    # Add to FAISS index
    index.add(embeddings)
    print(f" Added {len(embeddings)} vectors from {f}")

# Save FAISS index
faiss.write_index(index, r"D:\UDN_Project\udn_faiss_test.index")
print(" Test index saved → D:\\UDN_Project\\udn_faiss_test.index")
