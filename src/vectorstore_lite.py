"""
Self-contained lite vector store for cloud deployment.
Everything is in FAISS index + SQLite (text included, no CSV dependency).
"""

import os
import sqlite3
import numpy as np
import faiss
from typing import List, Dict, Any


class LiteVectorStore:
    """Self-contained FAISS + SQLite store (no external CSV files needed)."""

    def __init__(self, index_path: str, db_path: str = None):
        self.index_path = index_path
        self.db_path = db_path or index_path.replace(".index", ".db")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index not found: {index_path}")
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        print(f"Loading lite index from {index_path}...")
        self.index = faiss.read_index(index_path)
        print(f"Lite index loaded: {self.index.ntotal:,} vectors")

    def _get_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def count(self) -> int:
        return self.index.ntotal if self.index else 0

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

        if valid_indices:
            conn = self._get_db()
            for score, idx in valid_indices:
                row = conn.execute(
                    "SELECT article_id, article_title, date, paper, chunk_text "
                    "FROM documents WHERE idx = ?", (int(idx),)
                ).fetchone()

                if row:
                    ids.append(f"doc_{idx}")
                    documents.append(row[4] or "")
                    metadatas.append({
                        "article_id": row[0],
                        "article_title": row[1],
                        "date": row[2],
                        "paper": row[3],
                    })
                    distances.append(1 - score)
            conn.close()

        return {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
            "distances": distances
        }
