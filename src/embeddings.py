from sentence_transformers import SentenceTransformer
import numpy as np, pandas as pd, faiss, os

model = SentenceTransformer('all-MiniLM-L6-v2')  # Free + small + accurate
dim = 384
index = faiss.IndexFlatL2(dim)

base_dir = r"D:\UDN_Project\chunked"
files = [f for f in os.listdir(base_dir) if f.endswith(".csv")]

for f in files:
    df = pd.read_csv(os.path.join(base_dir, f))
    embeddings = model.encode(df['chunk_text'].tolist(), show_progress_bar=True)
    index.add(embeddings)

faiss.write_index(index, r"D:\UDN_Project\udn_faiss.index")
