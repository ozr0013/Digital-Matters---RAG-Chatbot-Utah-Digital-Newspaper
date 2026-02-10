"""
RAG Chatbot Module
Retrieval-Augmented Generation chatbot for Utah Digital Newspapers Archive.
"""

from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
from vectorstore import VectorStore


class RAGChatbot:
    """
    RAG-based chatbot that retrieves relevant newspaper articles
    and returns them as search results with citations.
    """

    def __init__(
        self,
        db_path: str,
        collection_name: str = "udn_archive",
        model_name: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the RAG chatbot.

        Args:
            db_path: Path to ChromaDB storage
            collection_name: Name of the ChromaDB collection
            model_name: Sentence transformer model for query embedding
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.vectorstore = VectorStore(db_path, collection_name)

        print(f"RAG Chatbot initialized with {self.vectorstore.count():,} documents")

    def query(
        self,
        user_question: str,
        top_k: int = 5,
        include_snippets: bool = True
    ) -> Dict[str, Any]:
        """
        Process a user query and return relevant results.

        Args:
            user_question: The user's search query
            top_k: Number of results to return
            include_snippets: Whether to include text snippets

        Returns:
            Dict with 'answer' (summary text) and 'sources' (list of citations)
        """
        if not user_question or not user_question.strip():
            return {
                "answer": "Please enter a search query.",
                "sources": []
            }

        # Embed the query
        query_embedding = self.model.encode(user_question).tolist()

        # Search ChromaDB
        results = self.vectorstore.search(query_embedding, n_results=top_k)

        if not results["ids"]:
            return {
                "answer": "No relevant articles found for your query. Try different keywords or a broader search term.",
                "sources": []
            }

        # Format the response
        sources = self._format_sources(results)
        answer = self._format_answer(user_question, results, sources)

        return {
            "answer": answer,
            "sources": sources
        }

    def _format_sources(self, results: Dict[str, Any]) -> List[Dict[str, str]]:
        """Format search results as source citations."""
        sources = []

        for i, (doc_id, document, metadata, distance) in enumerate(zip(
            results["ids"],
            results["documents"],
            results["metadatas"],
            results["distances"]
        )):
            # Extract metadata
            title = metadata.get("article_title", "")
            if not title or title == "nan" or title == "None":
                title = "Untitled Article"

            date = metadata.get("date", "")
            if date and "T" in date:
                date = date.split("T")[0]  # Remove time portion

            paper = metadata.get("paper", "Unknown Paper")
            article_id = metadata.get("article_id", "")

            # Create snippet (first 200 chars of document)
            snippet = document[:300] + "..." if len(document) > 300 else document

            # Calculate relevance score (convert distance to similarity)
            # Cosine distance: 0 = identical, 2 = opposite
            similarity = max(0, 1 - distance) * 100

            sources.append({
                "title": title,
                "snippet": snippet,
                "date": date,
                "paper": paper,
                "article_id": article_id,
                "relevance": f"{similarity:.0f}%"
            })

        return sources

    def _format_answer(
        self,
        question: str,
        results: Dict[str, Any],
        sources: List[Dict[str, str]]
    ) -> str:
        """Generate an answer summary based on search results."""
        num_results = len(sources)

        if num_results == 0:
            return "No relevant articles found."

        # Get unique papers and date range
        papers = set(s["paper"] for s in sources if s["paper"])
        dates = [s["date"] for s in sources if s["date"]]

        # Build answer
        answer_parts = [
            f"Found {num_results} relevant article{'s' if num_results > 1 else ''} from the Utah Digital Newspapers archive."
        ]

        if papers:
            paper_list = ", ".join(sorted(papers)[:3])
            if len(papers) > 3:
                paper_list += f" and {len(papers) - 3} more"
            answer_parts.append(f"Sources include: {paper_list}.")

        if dates:
            sorted_dates = sorted(dates)
            if len(sorted_dates) > 1 and sorted_dates[0] != sorted_dates[-1]:
                answer_parts.append(f"Date range: {sorted_dates[0]} to {sorted_dates[-1]}.")
            elif sorted_dates:
                answer_parts.append(f"Date: {sorted_dates[0]}.")

        # Add note about viewing sources
        answer_parts.append("See the sources below for detailed excerpts from each article.")

        return " ".join(answer_parts)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            "total_documents": self.vectorstore.count(),
            "model": self.model_name,
            "collection": self.vectorstore.collection_name
        }


# For testing
if __name__ == "__main__":
    # Test the chatbot
    CHROMA_PATH = r"E:\UDN_Project\chroma_db"

    print("Initializing chatbot...")
    chatbot = RAGChatbot(CHROMA_PATH)

    print("\nStats:", chatbot.get_stats())

    # Test query
    test_query = "women's suffrage in Utah"
    print(f"\nTest query: {test_query}")

    result = chatbot.query(test_query, top_k=3)
    print(f"\nAnswer: {result['answer']}")
    print(f"\nSources ({len(result['sources'])}):")
    for i, source in enumerate(result['sources'], 1):
        print(f"  {i}. {source['title']} ({source['paper']}, {source['date']})")
        print(f"     Relevance: {source['relevance']}")
        print(f"     Snippet: {source['snippet'][:100]}...")
