from src.pdf_loader import load_pdfs
from src.chunking import chunk_documents

documents = load_pdfs()
chunks = chunk_documents(
    documents,
    chunk_size=500,
    chunk_overlap=50,
)

print(f"Total pages: {len(documents)}")
print(f"Total chunks: {len(chunks)}")
print("\nFirst chunk:")
print(chunks[0])