"""
Vector Store Module
Provides interface to ChromaDB for storing and querying document embeddings.
"""

import os
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional

# Disable telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"


class VectorStore:
    """ChromaDB wrapper for document storage and semantic search."""

    def __init__(self, db_path: str, collection_name: str = "udn_archive"):
        """
        Initialize the vector store.

        Args:
            db_path: Path to ChromaDB persistent storage
            collection_name: Name of the collection to use
        """
        self.db_path = db_path
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self.collection.count()

    def add_documents(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        batch_size: int = 2000
    ) -> int:
        """
        Add documents to the collection in batches.

        Args:
            ids: Unique identifiers for each document
            embeddings: Pre-computed embedding vectors
            documents: Document texts
            metadatas: Metadata dicts for each document
            batch_size: Number of documents per batch

        Returns:
            Number of documents added
        """
        added = 0
        for i in range(0, len(ids), batch_size):
            j = min(i + batch_size, len(ids))
            self.collection.add(
                ids=ids[i:j],
                embeddings=embeddings[i:j],
                documents=documents[i:j],
                metadatas=metadatas[i:j],
            )
            added += (j - i)
        return added

    def search(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query vector (384-dim for all-MiniLM-L6-v2)
            n_results: Number of results to return
            where: Optional metadata filter
            where_document: Optional document content filter

        Returns:
            Dict with keys: ids, documents, metadatas, distances
        """
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"]
        }

        if where:
            kwargs["where"] = where
        if where_document:
            kwargs["where_document"] = where_document

        results = self.collection.query(**kwargs)

        # Flatten the results (query returns nested lists)
        return {
            "ids": results["ids"][0] if results["ids"] else [],
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else []
        }

    def get_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a single document by ID."""
        result = self.collection.get(ids=[doc_id], include=["documents", "metadatas"])
        if result["ids"]:
            return {
                "id": result["ids"][0],
                "document": result["documents"][0] if result["documents"] else None,
                "metadata": result["metadatas"][0] if result["metadatas"] else None
            }
        return None

    def delete_collection(self):
        """Delete the entire collection. Use with caution!"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
