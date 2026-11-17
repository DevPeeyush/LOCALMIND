
from offline_ai_assistant.rag.loader import DocumentLoader
from offline_ai_assistant.rag.chunker import DocumentChunker
from offline_ai_assistant.rag.embeddings import EmbeddingService
from offline_ai_assistant.rag.vector_store import VectorStore


def test_vector_store():

    # Step 1: Load documents
    loader = DocumentLoader()
    documents = loader.load_directory("data/documents")

    print(f"\nLoaded documents: {len(documents)}")

    # Step 2: Split documents
    chunker = DocumentChunker(
        chunk_size=500,
        chunk_overlap=100,
    )

    chunks = chunker.split_documents(documents)

    print(f"Generated chunks: {len(chunks)}")

    # Step 3: Initialize embeddings
    embedding_service = EmbeddingService()

    # Step 4: Initialize vector store
    vector_store = VectorStore(
        embedding_service=embedding_service,
        index_path="data/vector_store",
    )

    # Step 5: Create FAISS index
    vector_store.create(chunks)

    print("\nFAISS index created successfully!")

    # Step 6: Save FAISS index
    vector_store.save()

    print("FAISS index saved successfully!")

    # Step 7: Load saved index
    vector_store = VectorStore(
        embedding_service=embedding_service,
        index_path="data/vector_store",
    )

    vector_store.load()

    print("FAISS index loaded successfully!")

    # Step 8: Search
    query = "What is this Offline AI Assistant project?"

    results = vector_store.similarity_search(
        query=query,
        k=3,
    )

    print("\nSearch Results:")

    for index, document in enumerate(results):

        print(f"\nResult {index + 1}")
        print("Content:", document.page_content)
        print("Metadata:", document.metadata)

    assert len(results) > 0

    print("\nFAISS vector store test successful!")


if __name__ == "__main__":
    test_vector_store()
