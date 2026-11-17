
from pathlib import Path
from typing import List

import pytesseract
from pdf2image import convert_from_path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_core.documents import Document


class DocumentLoader:
    """
    Load PDF, TXT, DOCX and Markdown documents.

    PDF processing:
    1. Extract text using PyPDFLoader.
    2. Use Tesseract OCR when native text is insufficient.
    3. Skip nearly empty OCR pages.
    4. Skip low-content OCR cover pages.
    5. Preserve page-level metadata.
    """

    SUPPORTED_EXTENSIONS = {
        ".pdf",
        ".txt",
        ".docx",
        ".md",
    }

    # Native PDF extraction threshold
    MIN_NATIVE_TEXT_CHARS = 30

    # OCR filtering thresholds
    MIN_OCR_TEXT_CHARS = 40
    MAX_COVER_PAGE_CHARS = 180

    # OCR configuration
    OCR_DPI = 250
    OCR_LANGUAGE = "eng"
    OCR_CONFIG = "--psm 6"

    def load_file(self, file_path: str | Path) -> List[Document]:
        """
        Load a single supported document.
        """

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Document not found: {file_path}"
            )

        extension = file_path.suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format: {extension}"
            )

        if extension == ".pdf":
            documents = self._load_pdf(file_path)

        elif extension == ".docx":
            loader = Docx2txtLoader(str(file_path))
            documents = loader.load()

        else:
            loader = TextLoader(
                str(file_path),
                encoding="utf-8",
            )
            documents = loader.load()

        # Preserve common metadata for every document.
        for document in documents:
            document.metadata["source"] = str(file_path)
            document.metadata["file_name"] = file_path.name

        return documents

    def _load_pdf(self, file_path: Path) -> List[Document]:
        """
        Extract PDF text with OCR fallback.

        Native extraction is preferred. OCR is used only when
        the extracted native text is insufficient.
        """

        loader = PyPDFLoader(str(file_path))
        documents = loader.load()

        processed_documents = []

        for document in documents:

            native_text = document.page_content.strip()

            # Case 1: Native PDF text is sufficient.
            if len(native_text) >= self.MIN_NATIVE_TEXT_CHARS:

                document.metadata["extraction_method"] = (
                    "native_text"
                )

            # Case 2: Use OCR when native extraction is insufficient.
            else:

                page_number = int(
                    document.metadata.get("page", 0)
                ) + 1

                try:
                    images = convert_from_path(
                        str(file_path),
                        first_page=page_number,
                        last_page=page_number,
                        dpi=self.OCR_DPI,
                        grayscale=True,
                    )

                    if not images:
                        raise RuntimeError(
                            f"Could not render page {page_number}"
                        )

                    ocr_text = pytesseract.image_to_string(
                        images[0],
                        lang=self.OCR_LANGUAGE,
                        config=self.OCR_CONFIG,
                    ).strip()

                    # Skip nearly empty OCR pages.
                    if len(ocr_text) < self.MIN_OCR_TEXT_CHARS:
                        print(
                            f"[SKIPPED] Nearly empty OCR page: "
                            f"{file_path.name}, page {page_number}"
                        )
                        continue

                    # Skip low-content OCR cover pages only.
                    if (
                        page_number == 1
                        and len(ocr_text) < self.MAX_COVER_PAGE_CHARS
                    ):
                        print(
                            f"[SKIPPED] Low-content OCR cover page: "
                            f"{file_path.name}"
                        )
                        continue

                    document.page_content = ocr_text

                    document.metadata["extraction_method"] = "ocr"

                except Exception as error:

                    raise RuntimeError(
                        f"OCR failed for {file_path.name}, "
                        f"page {page_number}: {error}"
                    ) from error

            # Preserve metadata.
            document.metadata["source"] = str(file_path)
            document.metadata["file_name"] = file_path.name

            processed_documents.append(document)

        return processed_documents

    def load_directory(
        self,
        directory_path: str | Path,
    ) -> List[Document]:
        """
        Recursively load all supported documents from a directory.
        """

        directory_path = Path(directory_path)

        if not directory_path.exists():
            raise FileNotFoundError(
                f"Directory not found: {directory_path}"
            )

        if not directory_path.is_dir():
            raise NotADirectoryError(
                f"Expected a directory: {directory_path}"
            )

        all_documents = []

        for file_path in sorted(directory_path.rglob("*")):

            if not file_path.is_file():
                continue

            if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue

            documents = self.load_file(file_path)

            all_documents.extend(documents)

        return all_documents
