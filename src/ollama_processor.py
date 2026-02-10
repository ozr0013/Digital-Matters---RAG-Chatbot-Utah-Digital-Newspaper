"""
LLM Processor for RAG results.
Supports Groq (free cloud API) and Ollama (local).
Cleans OCR text, reasons about queries, and generates intelligent answers.
"""

import os
from typing import Dict, Any, List, Optional

# Try Groq first (free cloud API, no install needed)
try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

# Try Ollama as fallback (local)
try:
    import requests as _requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class LLMProcessor:
    """Process RAG results through an LLM for intelligent answers."""

    def __init__(
        self,
        backend: str = "groq",
        groq_api_key: str = None,
        groq_model: str = "llama-3.3-70b-versatile",
        ollama_model: str = "llama3.2",
        ollama_url: str = "http://localhost:11434"
    ):
        self.backend = backend
        self.groq_client = None
        self.groq_model = groq_model
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url

        if backend == "groq" and HAS_GROQ:
            key = groq_api_key or os.environ.get("GROQ_API_KEY", "")
            if key:
                self.groq_client = Groq(api_key=key)

    def is_available(self) -> bool:
        if self.backend == "groq":
            return self.groq_client is not None
        elif self.backend == "ollama" and HAS_REQUESTS:
            try:
                resp = _requests.get(f"{self.ollama_url}/api/tags", timeout=3)
                return resp.status_code == 200
            except:
                return False
        return False

    def _call_llm(self, system: str, prompt: str) -> str:
        """Call the LLM backend."""
        if self.backend == "groq" and self.groq_client:
            return self._call_groq(system, prompt)
        elif self.backend == "ollama":
            return self._call_ollama(system, prompt)
        return ""

    def _call_groq(self, system: str, prompt: str) -> str:
        try:
            response = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=600
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Groq error: {e}")
            return ""

    def _call_ollama(self, system: str, prompt: str) -> str:
        try:
            resp = _requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "system": system,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 600}
                },
                timeout=60
            )
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
        except Exception as e:
            print(f"Ollama error: {e}")
        return ""

    def process_rag_response(
        self,
        query: str,
        rag_response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process RAG results through LLM for intelligent answer."""
        result = rag_response.copy()
        sources = result.get("sources", [])

        if not sources:
            return result

        # Build context from retrieved documents
        context_parts = []
        for i, src in enumerate(sources[:5], 1):
            title = src.get("title", "Untitled")
            paper = src.get("paper", "Unknown")
            date = src.get("date", "Unknown date")
            snippet = src.get("snippet", "")[:500]
            context_parts.append(
                f"Source {i}: {title}\n"
                f"Paper: {paper} | Date: {date}\n"
                f"Text: {snippet}\n"
            )

        context = "\n---\n".join(context_parts)

        system = """You are a knowledgeable historical research assistant specializing in Utah history and the Utah Digital Newspapers archive.

Your job:
1. Read the user's question and the retrieved newspaper excerpts
2. Synthesize a clear, informative answer based ONLY on what the sources say
3. Note important details: dates, people, places, events
4. Acknowledge that text may have OCR errors from scanning old newspapers
5. If the sources don't clearly answer the question, say so honestly

Rules:
- Be concise (3-5 sentences max for the summary)
- Only state facts found in the sources - do NOT make things up
- Reference which source(s) support your points
- If text is garbled from OCR, interpret what you can and note the limitation"""

        prompt = f"""User question: "{query}"

Here are the most relevant newspaper excerpts found in the archive:

{context}

Based on these historical sources, provide a clear and informative answer to the user's question:"""

        llm_answer = self._call_llm(system, prompt)

        if llm_answer:
            result["answer"] = llm_answer
            result["llm_summary"] = llm_answer

        return result


# Backward-compatible alias
OllamaProcessor = LLMProcessor


if __name__ == "__main__":
    # Test with Groq
    key = os.environ.get("GROQ_API_KEY", "")
    if key:
        proc = LLMProcessor(backend="groq", groq_api_key=key)
        print(f"Groq available: {proc.is_available()}")
        if proc.is_available():
            test_response = {
                "answer": "Found 1 article",
                "sources": [{
                    "title": "Woman's Political Rights",
                    "paper": "Deseret News",
                    "date": "1883-01-31",
                    "snippet": "WOMANS POLITICAL rights THE woman suffrage convention at washington DC adopted a resolution denouncing the proposition",
                    "relevance": "74%"
                }]
            }
            result = proc.process_rag_response("women's rights in Utah", test_response)
            print(f"\nAnswer: {result['answer']}")
    else:
        print("Set GROQ_API_KEY to test. Get free key at: https://console.groq.com")
