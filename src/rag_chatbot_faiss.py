"""
RAG Chatbot using FAISS
No migration needed - loads directly from .npy and .csv files.
"""

from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from vectorstore_faiss import FAISSVectorStore


# Utah Digital Newspapers URL pattern
UDN_BASE_URL = "https://newspapers.lib.utah.edu/details?id="


class RAGChatbot:
    """RAG chatbot using FAISS for vector search."""

    def __init__(
        self,
        emb_dir: str,
        chunk_dir: str,
        index_path: str = None,
        model_name: str = "all-MiniLM-L6-v2",
        max_files: int = None
    ):
        """
        Initialize the chatbot.

        Args:
            emb_dir: Directory with .npy embedding files
            chunk_dir: Directory with .csv chunk files
            index_path: Path to save/load FAISS index (speeds up subsequent loads)
            model_name: Sentence transformer model
            max_files: Limit files to load (for testing)
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.vectorstore = FAISSVectorStore(emb_dir, chunk_dir, index_path, max_files)

        print(f"RAG Chatbot ready with {self.vectorstore.count():,} documents")

    def query(self, user_question: str, top_k: int = 5) -> Dict[str, Any]:
        """Process a query and return results."""
        if not user_question or not user_question.strip():
            return {"answer": "Please enter a search query.", "sources": []}

        # Embed query
        query_embedding = self.model.encode(user_question).tolist()

        # Search
        results = self.vectorstore.search(query_embedding, n_results=top_k)

        if not results["ids"]:
            return {
                "answer": "No relevant articles found.",
                "sources": []
            }

        # Format response
        sources = self._format_sources(results)
        answer = self._format_answer(user_question, sources)

        return {"answer": answer, "sources": sources}

    def _format_sources(self, results: Dict[str, Any]) -> List[Dict[str, str]]:
        """Format results as source citations."""
        sources = []

        for doc, meta, dist in zip(
            results["documents"],
            results["metadatas"],
            results["distances"]
        ):
            title = meta.get("article_title", "")
            if not title or title in ("nan", "None", ""):
                title = "Untitled Article"

            date = meta.get("date", "")
            if date and "T" in date:
                date = date.split("T")[0]

            paper = meta.get("paper", "Unknown Paper")
            article_id = meta.get("article_id", "")

            snippet = doc[:300] + "..." if len(doc) > 300 else doc
            similarity = max(0, 1 - dist) * 100

            # Build link to original article
            link = f"{UDN_BASE_URL}{article_id}" if article_id else ""

            sources.append({
                "title": title,
                "snippet": snippet,
                "date": date,
                "paper": paper,
                "article_id": article_id,
                "link": link,
                "relevance": f"{similarity:.0f}%"
            })

        return sources

    def _format_answer(self, question: str, sources: List[Dict]) -> str:
        """Generate answer summary."""
        n = len(sources)
        if n == 0:
            return "No relevant articles found."

        papers = set(s["paper"] for s in sources if s["paper"])
        dates = [s["date"] for s in sources if s["date"]]

        parts = [f"Found {n} relevant article{'s' if n > 1 else ''} from the Utah Digital Newspapers archive."]

        if papers:
            paper_list = ", ".join(sorted(papers)[:3])
            if len(papers) > 3:
                paper_list += f" and {len(papers) - 3} more"
            parts.append(f"Sources include: {paper_list}.")

        if dates:
            sorted_dates = sorted(dates)
            if len(sorted_dates) > 1 and sorted_dates[0] != sorted_dates[-1]:
                parts.append(f"Date range: {sorted_dates[0]} to {sorted_dates[-1]}.")

        parts.append("See the sources below for detailed excerpts.")
        return " ".join(parts)

    def get_stats(self) -> Dict[str, Any]:
        """Get stats about the vectorstore."""
        return {
            "total_documents": self.vectorstore.count(),
            "model": self.model_name,
            "backend": "FAISS"
        }


if __name__ == "__main__":
    EMB_DIR = r"E:\UDN_Project\embeddings"
    CHUNK_DIR = r"E:\UDN_Project\chunked"
    INDEX_PATH = r"E:\UDN_Project\faiss_index\udn.index"

    # Test with limited files first
    print("Initializing chatbot (loading 100 files for test)...")
    chatbot = RAGChatbot(EMB_DIR, CHUNK_DIR, INDEX_PATH, max_files=100)

    print("\nStats:", chatbot.get_stats())

    # Test query
    test = "women's suffrage Utah"
    print(f"\nQuery: {test}")
    result = chatbot.query(test, top_k=3)
    print(f"Answer: {result['answer']}")
    for i, src in enumerate(result['sources'], 1):
        print(f"  {i}. {src['title']} ({src['paper']}, {src['date']}) - {src['relevance']}")
