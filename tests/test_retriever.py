
from offline_ai_assistant.rag.embeddings import EmbeddingService
from offline_ai_assistant.rag.vector_store import VectorStore
from offline_ai_assistant.rag.retriever import Retriever


def test_retriever():

    # Step 1: Initialize embedding service
    embedding_service = EmbeddingService()

    # Step 2: Initialize vector store
    vector_store = VectorStore(
        embedding_service=embedding_service,
        index_path="data/vector_store",
    )

    # Step 3: Load existing FAISS index
    vector_store.load()

    print("\nFAISS index loaded successfully!")

    # Step 4: Initialize retriever
    retriever = Retriever(
        vector_store=vector_store,
        top_k=3,
    )

    # Step 5: Search query
    query = "What is this Offline AI Assistant project?"

    # Step 6: Retrieve documents
    results = retriever.retrieve(query)

    print("\nRetrieved Documents:")

    for index, document in enumerate(results):

        print(f"\nResult {index + 1}")
        print("Content:", document.page_content)
        print("Metadata:", document.metadata)

    # Step 7: Retrieve documents with scores
    scored_results = retriever.retrieve_with_scores(query)

    print("\nRetrieval Results with Scores:")

    for index, (document, score) in enumerate(scored_results):

        print(f"\nResult {index + 1}")
        print("Distance Score:", score)
        print("Content:", document.page_content)

    assert len(results) > 0
    assert len(scored_results) > 0

    print("\nRetriever test successful!")


if __name__ == "__main__":
    test_retriever()
