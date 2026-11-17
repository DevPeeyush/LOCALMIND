
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunker:
    """Split documents into smaller chunks for RAG."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

    def split_documents(
        self,
        documents: List[Document],
    ) -> List[Document]:
        """Split loaded documents into chunks."""

        if not documents:
            return []

        chunks = self.splitter.split_documents(documents)

        return chunks
