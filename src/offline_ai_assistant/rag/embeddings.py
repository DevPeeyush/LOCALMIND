
from typing import List

from langchain_ollama import OllamaEmbeddings


class EmbeddingService:
    """Generate embeddings using a local Ollama model."""

    MODEL_NAME = "nomic-embed-text"

    def __init__(self):

        self.model = OllamaEmbeddings(
            model=self.MODEL_NAME,
            base_url="http://localhost:11434",
        )

    def embed_documents(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """Generate embeddings for documents."""

        return self.model.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a user query."""

        return self.model.embed_query(text)
