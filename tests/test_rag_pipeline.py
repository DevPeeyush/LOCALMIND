
from offline_ai_assistant.rag.embeddings import EmbeddingService
from offline_ai_assistant.rag.vector_store import VectorStore
from offline_ai_assistant.rag.retriever import Retriever
from offline_ai_assistant.rag.pipeline import RAGPipeline


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

    rag = RAGPipeline(
        retriever=retriever,
    )

    test_questions = [
        "Does Share India provide medical insurance?",
        "What digital healthcare facilities are available?",
        "What employee benefits does Share India provide?",
        "What is the employee's individual insurance policy number?",
    ]

    for question in test_questions:

        print("\n" + "=" * 90)
        print("QUESTION:", question)
        print("=" * 90)

        try:
            result = rag.run(
                query=question,
                k=5,
            )

            print("\nANSWER:")
            print(result["answer"])

            print("\nSOURCES:")

            for source in result["sources"]:
                print(
                    f"[{source['source_id']}] "
                    f"{source['file_name']} | "
                    f"Page: {source['page']}"
                )

        except Exception as error:
            print("\nERROR:", str(error))


if __name__ == "__main__":
    main()
