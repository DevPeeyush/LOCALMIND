
from pathlib import Path

from offline_ai_assistant.rag.loader import DocumentLoader


def test_document_loader():

    loader = DocumentLoader()

    documents_path = Path("data/documents")

    documents_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    test_file = documents_path / "sample.txt"

    test_file.write_text(
        "This is a test document for our Offline AI Assistant RAG pipeline.",
        encoding="utf-8",
    )

    documents = loader.load_file(test_file)

    assert len(documents) > 0
    assert "Offline AI Assistant" in documents[0].page_content

    print("Document loading successful!")
    print("Content:", documents[0].page_content)
    print("Metadata:", documents[0].metadata)


if __name__ == "__main__":
    test_document_loader()
