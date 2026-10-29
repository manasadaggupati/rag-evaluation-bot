
import json
from pathlib import Path
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
LABELS_FILE = ROOT / "evaluation" / "relevance_labels.json"
PDF_FOLDER = ROOT / "data" / "documents"

labels = json.loads(LABELS_FILE.read_text(encoding="utf-8"))

for item in labels:
    print("\n" + "=" * 80)
    print("QUESTION:", item["question"])

    for passage in item["relevant_passages"]:
        pdf_path = PDF_FOLDER / passage["source"]
        page_number = int(passage["page"])

        reader = PdfReader(str(pdf_path))

        print("\n" + "-" * 60)
        print(f"Source: {passage['source']} | Page: {page_number}")
        print("Reason:", passage["reason"])

        if 1 <= page_number <= len(reader.pages):
            text = reader.pages[page_number - 1].extract_text() or ""
            print("\nACTUAL PDF TEXT:\n")
            print(text[:1800])
        else:
            print("ERROR: Page number is outside the PDF's page range.")
