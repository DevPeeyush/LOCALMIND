
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
        top_k=10,
    )

    queries = [
        "What employee benefits does Share India provide?",
        "Does Share India provide medical insurance?",
        "What healthcare, food, transport and wellness facilities are provided?",
    ]

    for query in queries:

        print("\n" + "=" * 85)
        print(f"QUESTION: {query}")
        print("=" * 85)

        print(
            "Broad benefits query:",
            retriever.is_broad_benefits_query(query),
        )

        results = retriever.retrieve_with_scores(
            query=query,
            k=10,
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
                "\n",
                " ",
            )

            print(
                f"\nRank: {rank} | "
                f"RRF Score: {score:.6f} | "
                f"File: {filename} | "
                f"Page: {page}"
            )

            print(f"Content: {content[:300]}")


if __name__ == "__main__":
    main()
