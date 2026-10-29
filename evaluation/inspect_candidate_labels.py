
import json
from pathlib import Path
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_FILE = ROOT / "evaluation" / "test_questions.json"
PDF_FOLDER = ROOT / "data" / "documents"

questions = json.loads(
    QUESTIONS_FILE.read_text(encoding="utf-8")
)

for item in questions:
    if "expected_page" not in item:
        continue

    pdf_path = PDF_FOLDER / item["expected_source"]
    page_number = int(item["expected_page"])

    reader = PdfReader(str(pdf_path))

    if not 1 <= page_number <= len(reader.pages):
        print(f"\nERROR: Invalid page {page_number}: {item['question']}")
        continue

    text = reader.pages[page_number - 1].extract_text() or ""

    print("\n" + "=" * 80)
    print("QUESTION:", item["question"])
    print("SOURCE:", item["expected_source"])
    print("PAGE:", page_number)
    print("\nPDF TEXT:\n")
    print(text[:1500])
