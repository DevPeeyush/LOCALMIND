
"""
Initialize and share application services.
"""

from pathlib import Path
from offline_ai_assistant.assistant.conversation_store import (
    ConversationStore,
)
from offline_ai_assistant.assistant.session_manager import (
    SessionManager,
)
from offline_ai_assistant.rag.embeddings import EmbeddingService
from offline_ai_assistant.rag.vector_store import VectorStore
from offline_ai_assistant.rag.retriever import Retriever
from offline_ai_assistant.rag.pipeline import RAGPipeline
from offline_ai_assistant.core.model_manager import ModelManager

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
DATABASE_PATH = DATA_DIR / "conversations.db"



def initialize_services() -> dict:
    # Initialize the shared chat model manager.
    model_manager = ModelManager()

    # Initialize fixed embedding service.
    embedding_service = EmbeddingService()

    # Load existing FAISS vector store.
    vector_store = VectorStore(
        embedding_service=embedding_service,
        index_path=str(VECTOR_STORE_DIR),
    )
    vector_store.load()

    # Initialize retrieval.
    retriever = Retriever(
        vector_store=vector_store,
        top_k=3,
    )

    # Initialize RAG with shared model manager.
    rag_pipeline = RAGPipeline(
        retriever=retriever,
        model_manager=model_manager,
    )

    # Initialize conversation storage.
    conversation_store = ConversationStore(
        db_path=DATABASE_PATH,
    )

    # Initialize session manager.
    session_manager = SessionManager(
    store=conversation_store,
    rag_pipeline=rag_pipeline,
    max_messages=20,
    model_manager=model_manager,
)

    return {
        "model_manager": model_manager,
        "embedding_service": embedding_service,
        "vector_store": vector_store,
        "retriever": retriever,
        "rag_pipeline": rag_pipeline,
        "conversation_store": conversation_store,
        "session_manager": session_manager,
    }

