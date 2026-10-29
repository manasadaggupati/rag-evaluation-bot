
from src.pdf_loader import load_pdfs

target_pages = {6, 7, 10, 12, 13, 14}

documents = load_pdfs()

for doc in documents:
    if doc["source"] == "RAG.pdf" and doc["page"] in target_pages:
        print("\n" + "=" * 80)
        print(f"FILE: {doc['source']} | PAGE: {doc['page']}")
        print("=" * 80)
        print(doc["text"])
