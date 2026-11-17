
from offline_ai_assistant.rag.embeddings import EmbeddingService
from offline_ai_assistant.rag.vector_store import VectorStore
from offline_ai_assistant.rag.retriever import Retriever


def inspect_index(vector_store):
    """Inspect all documents stored in the FAISS index."""

    all_documents = list(
        vector_store.vector_store.docstore._dict.values()
    )

    print("\n" + "=" * 70)
    print("INDEXED DOCUMENTS")
    print("=" * 70)

    indexed_files = set()

    for document in all_documents:
        filename = document.metadata.get("file_name")
        source = document.metadata.get("source")

        indexed_files.add(filename or source or "Unknown")

    for filename in sorted(indexed_files):
        print(filename)

    print(f"\nTotal indexed chunks: {len(all_documents)}")
    print(f"Total unique documents: {len(indexed_files)}")

    # Inspect the two previously missing documents
    target_files = [
        "employee_benefits.pdf",
        "posh_policy.pdf",
    ]

    for target_file in target_files:

        matching_documents = [
            document
            for document in all_documents
            if (
                document.metadata.get("file_name") == target_file
                or target_file in str(
                    document.metadata.get("source", "")
                )
            )
        ]

        print("\n" + "=" * 70)
        print(f"INSPECTION: {target_file}")
        print("=" * 70)

        print(f"Indexed chunks found: {len(matching_documents)}")

        for index, document in enumerate(
            matching_documents, start=1
        ):
            print(f"\nChunk {index}")

            print(
                "Page:",
                document.metadata.get(
                    "page_label",
                    document.metadata.get("page", "Unknown"),
                ),
            )

            print("Content:", document.page_content[:400])

    return all_documents


def run_retrieval_tests(retriever):
    """Test retrieval using employee-policy questions."""

    queries = [
        "What employee benefits are available?",
        "What employee benefits does Share India Securities provide?",
        "What medical insurance and meal facilities are available to employees?",
        "Does Share India provide medical insurance, breakfast, lunch, and evening snacks?",
        "What facilities and benefits does the company offer its employees?",
    ]

    for query in queries:

        print("\n" + "=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)

        results = retriever.retrieve_with_scores(
            query=query,
            k=8,
        )

        if not results:
            print("No relevant documents found.")
            continue

        for index, (document, distance) in enumerate(
            results, start=1
        ):

            print(f"\nResult {index}")
            print(f"Distance: {distance:.4f}")

            print(
                "Document:",
                document.metadata.get("file_name", "Unknown"),
            )

            print(
                "Page:",
                document.metadata.get(
                    "page_label",
                    document.metadata.get("page", "Unknown"),
                ),
            )

            print(
                "Content:",
                document.page_content[:300],
            )


def main():

    # Step 1: Initialize embedding service
    print("\nInitializing embedding service...")

    embedding_service = EmbeddingService()

    # Step 2: Initialize vector store
    print("Initializing vector store...")

    vector_store = VectorStore(
        embedding_service=embedding_service,
        index_path="data/vector_store",
    )

    # Step 3: Load the exact index created by test_vector_store.py
    print("Loading FAISS index...")

    vector_store.load()

    print("FAISS index loaded successfully!")

    # Step 4: Initialize retriever
    retriever = Retriever(
        vector_store=vector_store,
        top_k=8,
    )

    # Step 5: Inspect indexed documents
    inspect_index(vector_store)

    # Step 6: Run retrieval tests
    run_retrieval_tests(retriever)


if __name__ == "__main__":
    main()
