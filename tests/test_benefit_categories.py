
from offline_ai_assistant.rag.embeddings import EmbeddingService
from offline_ai_assistant.rag.vector_store import VectorStore
from offline_ai_assistant.rag.retriever import Retriever


def main():

    print("\nInitializing embedding service...")
    embedding_service = EmbeddingService()

    print("Loading FAISS index...")
    vector_store = VectorStore(
        embedding_service=embedding_service,
        index_path="data/vector_store",
    )
    vector_store.load()

    retriever = Retriever(
        vector_store=vector_store,
        top_k=5,
    )

    test_queries = [
        "Does Share India provide medical insurance?",
        "What digital healthcare facilities are available?",
        "Does Share India provide subsidized meals?",
        "Does Share India provide free employee transport?",
        "Does Share India organize corporate outings?",
        "What employee engagement activities are organized?",
        "Does Share India provide yoga and wellness facilities?",
    ]

    for query in test_queries:

        print("\n" + "=" * 90)
        print("QUESTION:", query)
        print("=" * 90)

        results = retriever.retrieve_with_scores(
            query=query,
            k=5,
        )

        for rank, (document, score) in enumerate(
            results,
            start=1,
        ):

            filename = document.metadata.get(
                "file_name",
                "Unknown",
            )

            page = document.metadata.get(
                "page_label",
                document.metadata.get("page", "Unknown"),
            )

            content = document.page_content.strip().replace(
                "\n", " "
            )

            print(
                f"\nRank: {rank} | "
                f"Score: {score:.6f} | "
                f"File: {filename} | "
                f"Page: {page}"
            )

            print("Content:", content[:250])


if __name__ == "__main__":
    main()
