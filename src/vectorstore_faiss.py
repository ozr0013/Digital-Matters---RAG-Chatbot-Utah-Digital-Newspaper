"""
FAISS-based Vector Store
- Small dataset (<200 files): IndexFlatIP (exact search)
- Full dataset (all files): IVF+PQ (compressed, handles 300M+ vectors)
- SQLite stores only metadata (not full text) to keep DB small
- Full text is looked up from CSV on-demand during search
"""

import os
import sqlite3
import numpy as np
import pandas as pd
import faiss
from typing import List, Dict, Any, Optional
import time


class FAISSVectorStore:
    """FAISS vector store with lightweight SQLite metadata."""

    def __init__(
        self,
        emb_dir: str,
        chunk_dir: str,
        index_path: Optional[str] = None,
        max_files: Optional[int] = None
    ):
        self.emb_dir = emb_dir
        self.chunk_dir = chunk_dir
        self.index_path = index_path
        self.max_files = max_files

        self.index = None
        self.db_path = index_path.replace(".index", ".db") if index_path else None

        if index_path and os.path.exists(index_path) and os.path.exists(self.db_path):
            self._load_index(index_path)
        else:
            self._build_index()
            if index_path:
                self._save_index(index_path)

    def _get_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".", exist_ok=True)
        conn = self._get_db()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                idx INTEGER PRIMARY KEY,
                article_id TEXT,
                article_title TEXT,
                date TEXT,
                paper TEXT,
                source_file TEXT,
                row_num INTEGER
            )
        """)
        conn.commit()
        conn.close()

    def _build_index(self):
        print("Building FAISS index...")

        files = sorted([f for f in os.listdir(self.emb_dir) if f.endswith(".npy")])
        if self.max_files:
            files = files[:self.max_files]

        total_files = len(files)
        print(f"Processing {total_files} files...")

        self._init_db()

        use_compressed = total_files > 200
        dim = 384

        if use_compressed:
            print("Mode: Compressed IVF+PQ (large dataset)")
            self._build_compressed(files, dim)
        else:
            print("Mode: Flat index (small dataset)")
            self._build_flat(files, dim)

    def _load_file_embeddings_and_meta(self, npy_file):
        """Load embeddings and lightweight metadata (no full text)."""
        base = npy_file.replace(".npy", "")
        csv_path = os.path.join(self.chunk_dir, f"{base}.csv")
        npy_path = os.path.join(self.emb_dir, npy_file)

        if not os.path.exists(csv_path):
            return None

        try:
            embeddings = np.load(npy_path).astype(np.float32)

            try:
                df = pd.read_csv(csv_path, dtype=str, usecols=lambda c: c != "chunk_text")
            except:
                df = pd.read_csv(csv_path, dtype=str, engine="python")

            if df.empty or len(df) != embeddings.shape[0]:
                return None

            rows = []
            for i, row in df.iterrows():
                rows.append({
                    "article_id": str(row.get("id", "") or ""),
                    "article_title": str(row.get("article_title", "") or ""),
                    "date": str(row.get("date", "") or ""),
                    "paper": str(row.get("paper", "") or ""),
                    "source_file": base,
                    "row_num": i
                })

            return embeddings, rows

        except Exception as e:
            print(f"  Error: {npy_file}: {e}")
            return None

    def _build_flat(self, files, dim):
        all_embeddings = []
        global_idx = 0
        conn = self._get_db()

        for file_idx, npy_file in enumerate(files):
            data = self._load_file_embeddings_and_meta(npy_file)
            if data is None:
                continue

            embeddings, rows = data
            all_embeddings.append(embeddings)

            batch = []
            for row in rows:
                batch.append((global_idx, row["article_id"], row["article_title"],
                              row["date"], row["paper"], row["source_file"], row["row_num"]))
                global_idx += 1

            conn.executemany("INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)", batch)

            if (file_idx + 1) % 10 == 0:
                conn.commit()
                print(f"  {file_idx + 1}/{len(files)} files, {global_idx:,} docs")

        conn.commit()
        conn.close()

        if not all_embeddings:
            raise ValueError("No embeddings loaded!")

        print("Creating flat index...")
        all_embeddings = np.vstack(all_embeddings)
        faiss.normalize_L2(all_embeddings)

        self.index = faiss.IndexFlatIP(dim)
        self.index.add(all_embeddings)
        print(f"Index built: {self.index.ntotal:,} vectors")

    def _build_compressed(self, files, dim):
        start = time.time()

        # Phase 1: Train on sample
        print("Phase 1: Training index...")
        training_data = []
        training_target = 500_000

        for npy_file in files[:30]:
            data = self._load_file_embeddings_and_meta(npy_file)
            if data is None:
                continue
            embeddings, _ = data
            training_data.append(embeddings)
            if sum(e.shape[0] for e in training_data) >= training_target:
                break

        training_vectors = np.vstack(training_data)
        faiss.normalize_L2(training_vectors)

        if training_vectors.shape[0] > training_target:
            idx = np.random.choice(training_vectors.shape[0], training_target, replace=False)
            training_vectors = training_vectors[idx]

        print(f"  Training on {training_vectors.shape[0]:,} samples")

        n_clusters = min(4096, training_vectors.shape[0] // 40)
        pq_subvectors = 48
        pq_bits = 8

        quantizer = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIVFPQ(quantizer, dim, n_clusters, pq_subvectors, pq_bits)
        self.index.train(training_vectors)
        print(f"  Trained with {n_clusters} clusters")

        del training_data, training_vectors

        # Phase 2: Add all vectors + metadata
        print("Phase 2: Adding vectors...")
        conn = self._get_db()
        global_idx = 0

        for file_idx, npy_file in enumerate(files):
            data = self._load_file_embeddings_and_meta(npy_file)
            if data is None:
                continue

            embeddings, rows = data
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)

            batch = []
            for row in rows:
                batch.append((global_idx, row["article_id"], row["article_title"],
                              row["date"], row["paper"], row["source_file"], row["row_num"]))
                global_idx += 1

            conn.executemany("INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)", batch)

            if (file_idx + 1) % 50 == 0:
                conn.commit()
                elapsed = (time.time() - start) / 60
                rate = global_idx / elapsed if elapsed > 0 else 0
                eta = (elapsed / (file_idx + 1)) * (len(files) - file_idx - 1)
                print(f"  [{file_idx + 1}/{len(files)}] {global_idx:,} docs | "
                      f"{rate:,.0f}/min | ETA: {eta:.0f} min")

        conn.commit()
        conn.close()

        self.index.nprobe = 32
        elapsed = (time.time() - start) / 60
        print(f"Index built: {self.index.ntotal:,} vectors in {elapsed:.1f} min")

    def _save_index(self, path: str):
        print(f"Saving index to {path}...")
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        faiss.write_index(self.index, path)
        print("Index saved!")

    def _load_index(self, path: str):
        print(f"Loading index from {path}...")
        self.index = faiss.read_index(path)
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = 32
        print(f"Index loaded: {self.index.ntotal:,} vectors")

    def count(self) -> int:
        return self.index.ntotal if self.index else 0

    def _lookup_text(self, source_file: str, row_num: int) -> str:
        """Look up full text from original CSV on-demand."""
        csv_path = os.path.join(self.chunk_dir, f"{source_file}.csv")
        if not os.path.exists(csv_path):
            return ""
        try:
            df = pd.read_csv(csv_path, dtype=str, skiprows=range(1, row_num + 1), nrows=1)
            if not df.empty and "chunk_text" in df.columns:
                return str(df["chunk_text"].iloc[0] or "")
        except:
            pass
        return ""

    def search(self, query_embedding: List[float], n_results: int = 5) -> Dict[str, Any]:
        """Search for similar documents."""
        query = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query)

        scores, indices = self.index.search(query, n_results)

        ids = []
        documents = []
        metadatas = []
        distances = []

        valid_indices = [(s, i) for s, i in zip(scores[0], indices[0]) if i >= 0]

        if valid_indices and self.db_path and os.path.exists(self.db_path):
            conn = self._get_db()
            for score, idx in valid_indices:
                row = conn.execute(
                    "SELECT article_id, article_title, date, paper, source_file, row_num "
                    "FROM documents WHERE idx = ?", (int(idx),)
                ).fetchone()

                if row:
                    # Look up full text from CSV
                    text = self._lookup_text(row[4], row[5])

                    ids.append(f"doc_{idx}")
                    documents.append(text)
                    metadatas.append({
                        "article_id": row[0],
                        "article_title": row[1],
                        "date": row[2],
                        "paper": row[3],
                        "chunk_index": str(row[5]),
                        "source_file": row[4]
                    })
                    distances.append(1 - score)
            conn.close()

        return {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
            "distances": distances
        }


if __name__ == "__main__":
    EMB_DIR = r"E:\UDN_Project\embeddings"
    CHUNK_DIR = r"E:\UDN_Project\chunked"
    INDEX_PATH = r"E:\UDN_Project\faiss_index\udn.index"

    store = FAISSVectorStore(EMB_DIR, CHUNK_DIR, INDEX_PATH, max_files=50)
    print(f"\nTotal documents: {store.count():,}")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    query = "Utah mining history"
    query_emb = model.encode(query).tolist()

    results = store.search(query_emb, n_results=3)
    print(f"\nSearch for '{query}':")
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"], results["metadatas"], results["distances"]
    ), 1):
        title = meta['article_title'][:50] if meta['article_title'] else 'Untitled'
        print(f"  {i}. {title} ({meta['paper']})")
        print(f"     {doc[:100]}...")
