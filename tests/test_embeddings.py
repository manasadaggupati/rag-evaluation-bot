from src.pdf_loader import load_pdfs
from src.chunking import chunk_documents
from src.embeddings import create_embeddings

documents = load_pdfs()
chunks = chunk_documents(documents, chunk_size=500, chunk_overlap=50)

texts = [chunk["text"] for chunk in chunks[:5]]
embeddings = create_embeddings(texts)

print("Chunks embedded:", len(embeddings))
print("Embedding dimensions:", embeddings.shape[1])
print("Vector example:", embeddings[0][:5])
