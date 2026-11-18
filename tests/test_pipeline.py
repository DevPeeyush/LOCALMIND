
from offline_ai_assistant.rag.embeddings import EmbeddingService
from offline_ai_assistant.rag.vector_store import VectorStore
from offline_ai_assistant.rag.retriever import Retriever
from offline_ai_assistant.rag.pipeline import RAGPipeline


def test_rag_pipeline():

    # 1. Initialize embedding service
    embedding_service = EmbeddingService()

    # 2. Initialize FAISS vector store
    vector_store = VectorStore(
        embedding_service=embedding_service,
        index_path="data/vector_store",
    )

    # 3. Load existing index
    vector_store.load()

    print("\nFAISS index loaded successfully!")

    # 4. Initialize retriever
    retriever = Retriever(
        vector_store=vector_store,
        top_k=3,
    )

    # 5. Initialize RAG pipeline
    rag_pipeline = RAGPipeline(
        retriever=retriever,
    )

    # 6. Ask HR policy question
    query = "What is the grievance redressal procedure for employees?"

    print("\nEmployee Question:")
    print(query)

    # 7. Execute RAG
    result = rag_pipeline.run(
        query=query,
        k=3,
    )

    # 8. Display answer
    print("\nGenerated Answer:")
    print(result["answer"])

    # 9. Display source information
    print("\nRetrieved Sources:")

    for source in result["sources"]:

        print(f"\nSource ID: {source['source_id']}")
        print(f"Document: {source['file_name']}")
        print(f"Page: {source['page']}")

    # 10. Validate response
    assert result["question"] == query
    assert isinstance(result["answer"], str)
    assert result["answer"].strip()
    assert isinstance(result["sources"], list)
    assert len(result["sources"]) > 0

    # Validate the actual public source response structure
    for source in result["sources"]:
        assert source["source_id"]
        assert source["source_id"].startswith("S")
        assert source["file_name"]
        assert "page" in source

    print("\nSource-aware RAG test successful!")


if __name__ == "__main__":
    test_rag_pipeline()
