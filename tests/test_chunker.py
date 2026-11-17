
from offline_ai_assistant.rag.loader import DocumentLoader
from offline_ai_assistant.rag.chunker import DocumentChunker


def test_document_chunker():

    loader = DocumentLoader()

    documents = loader.load_directory("data/documents")

    chunker = DocumentChunker(
        chunk_size=500,
        chunk_overlap=100,
    )

    chunks = chunker.split_documents(documents)

    print(f"\nTotal documents: {len(documents)}")
    print(f"Total chunks: {len(chunks)}")

    for index, chunk in enumerate(chunks):

        print(f"\n--- Chunk {index + 1} ---")
        print("Content:", chunk.page_content)
        print("Metadata:", chunk.metadata)

    assert len(chunks) > 0

    print("\nChunking successful!")


if __name__ == "__main__":
    test_document_chunker()
