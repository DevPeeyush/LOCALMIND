
"""
Document management API.

Supports:
- PDF upload and ingestion.
- Listing indexed documents.
- Retrieving document details.
- Deleting documents and rebuilding the FAISS index.
"""

from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from offline_ai_assistant.api.dependencies import PROJECT_ROOT
from offline_ai_assistant.rag.loader import DocumentLoader
from offline_ai_assistant.rag.chunker import DocumentChunker

router = APIRouter(
    prefix="/api/v1/documents",
    tags=["Documents"],
)

DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
UPLOAD_DIR = DOCUMENTS_DIR / "uploads"

MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB


def get_document_files():
    """Return all supported documents in the document directory."""
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    return sorted(
        path
        for path in DOCUMENTS_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in DocumentLoader.SUPPORTED_EXTENSIONS
    )


def find_document(document_id: str) -> Path:
    """Find a document by its unique filename."""
    if Path(document_id).name != document_id:
        raise HTTPException(status_code=400, detail="Invalid document ID.")

    for path in get_document_files():
        if path.name == document_id:
            return path

    raise HTTPException(status_code=404, detail="Document not found.")


@router.post("/upload")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
):
    """Upload and index a PDF document."""

    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    if Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported by this upload endpoint.",
        )

    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File exceeds the 25 MB upload limit.",
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    safe_name = Path(file.filename).name
    document_id = f"{uuid4().hex}_{safe_name}"
    file_path = UPLOAD_DIR / document_id

    loader = DocumentLoader()
    chunker = DocumentChunker()

    vector_store = request.app.state.vector_store

    try:
        file_path.write_bytes(content)

        documents = loader.load_file(file_path)
        chunks = chunker.split_documents(documents)

        if not chunks:
            raise ValueError("No readable text was found in the PDF.")

        if vector_store.vector_store is None:
            raise RuntimeError("FAISS index is not initialized.")

        # Add new chunks without replacing the existing index.
        vector_store.vector_store.add_documents(chunks)
        vector_store.save()

        return {
            "success": True,
            "message": "Document uploaded and indexed successfully.",
            "document_id": document_id,
            "filename": safe_name,
            "pages": len(documents),
            "chunks": len(chunks),
        }

    except Exception as error:
        file_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail=f"Document ingestion failed: {str(error)}",
        ) from error

    finally:
        await file.close()


@router.get("/")
def list_documents():
    """List documents stored in the knowledge base."""

    files = get_document_files()

    return {
        "total_documents": len(files),
        "documents": [
            {
                "document_id": path.name,
                "filename": path.name,
                "file_type": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
                "location": str(path.relative_to(DOCUMENTS_DIR)),
            }
            for path in files
        ],
    }


@router.get("/{document_id}")
def get_document(document_id: str):
    """Get metadata for one document."""

    path = find_document(document_id)

    return {
        "document_id": path.name,
        "filename": path.name,
        "file_type": path.suffix.lower(),
        "size_bytes": path.stat().st_size,
        "location": str(path.relative_to(DOCUMENTS_DIR)),
    }


@router.delete("/{document_id}")
def delete_document(request: Request, document_id: str):
    """Delete a document and rebuild the FAISS index."""

    file_path = find_document(document_id)

    loader = DocumentLoader()
    chunker = DocumentChunker()
    vector_store = request.app.state.vector_store

    # Keep a backup until the new index has been saved.
    backup_path = file_path.with_name(
        f"{file_path.name}.deleting"
    )

    try:
        file_path.rename(backup_path)

        remaining_documents = loader.load_directory(DOCUMENTS_DIR)
        remaining_chunks = chunker.split_documents(remaining_documents)

        if not remaining_chunks:
            raise ValueError(
                "Cannot delete the last indexed document."
            )

        # Rebuild so deleted chunks cannot remain searchable.
        vector_store.create(remaining_chunks)
        vector_store.save()

        backup_path.unlink(missing_ok=True)

        return {
            "success": True,
            "message": "Document deleted and FAISS index rebuilt.",
            "document_id": document_id,
            "remaining_chunks": len(remaining_chunks),
        }

    except Exception as error:
        # Restore the original file if rebuilding fails.
        if backup_path.exists():
            backup_path.rename(file_path)

        raise HTTPException(
            status_code=500,
            detail=f"Document deletion failed: {str(error)}",
        ) from error
