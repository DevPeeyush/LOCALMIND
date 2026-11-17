
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from offline_ai_assistant.rag.embeddings import EmbeddingService


class VectorStore:
    """Manage local FAISS vector storage."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        index_path: str = "data/vector_store",
    ):
        self.embedding_service = embedding_service
        self.index_path = Path(index_path)
        self.vector_store: Optional[FAISS] = None

    def create(
        self,
        documents: List[Document],
    ) -> FAISS:
        """Create a FAISS index from document chunks."""

        if not documents:
            raise ValueError("Cannot create index from empty documents.")

        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embedding_service.model,
        )

        return self.vector_store

    def save(self) -> None:
        """Save FAISS index to local storage."""

        if self.vector_store is None:
            raise ValueError("No FAISS index available to save.")

        self.index_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.vector_store.save_local(
            str(self.index_path)
        )

    def load(self) -> FAISS:
        """Load an existing FAISS index."""

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {self.index_path}"
            )

        self.vector_store = FAISS.load_local(
            str(self.index_path),
            self.embedding_service.model,
            allow_dangerous_deserialization=True,
        )

        return self.vector_store

    def similarity_search(
        self,
        query: str,
        k: int = 3,
    ) -> List[Document]:
        """Retrieve the most similar document chunks."""

        if self.vector_store is None:
            raise ValueError(
                "FAISS index is not initialized. Create or load an index first."
            )

        if not query.strip():
            raise ValueError("Search query cannot be empty.")

        if k <= 0:
            raise ValueError("k must be positive.")

        return self.vector_store.similarity_search(
            query,
            k=k,
        )
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 3,
    ):
        """Retrieve documents with their FAISS distance scores."""

        if self.vector_store is None:
            raise ValueError(
                "FAISS index is not initialized. Create or load an index first."
            )

        if not query.strip():
            raise ValueError("Search query cannot be empty.")

        if k <= 0:
            raise ValueError("k must be positive.")

        return self.vector_store.similarity_search_with_score(
            query,
            k=k,
        )

