from pathlib import Path
from pypdf import PdfReader


def load_pdfs(folder_path: str = "data/documents") -> list[dict]:
    documents = []

    for pdf_path in sorted(Path(folder_path).glob("*.pdf")):
        reader = PdfReader(str(pdf_path))

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            if text.strip():
                documents.append({
                    "source": pdf_path.name,
                    "page": page_number,
                    "text": text,
                })

    return documents


if __name__ == "__main__":
    documents = load_pdfs()

    print(f"Extracted {len(documents)} pages with text.")

    for document in documents:
        print(
            f"{document['source']} | "
            f"Page {document['page']} | "
            f"{len(document['text'])} characters"
        )