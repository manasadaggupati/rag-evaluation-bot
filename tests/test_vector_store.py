
from src.pdf_loader import load_pdfs
from src.chunking import chunk_documents
from src.vector_store import index_chunks, search_chunks


def test_vector_store_retrieves_relevant_documents():
    documents = load_pdfs()
    chunks = chunk_documents(
        documents,
        chunk_size=500,
        chunk_overlap=50,
    )

    assert documents, "No PDF documents were loaded."
    assert chunks, "No chunks were generated."

    indexed_count = index_chunks(chunks)
    assert indexed_count == len(chunks)

    results = search_chunks(
        "What is the Central Limit Theorem?",
        top_k=5,
    )

    assert results["documents"], "No documents were returned."
    assert results["documents"][0], "No relevant chunks were retrieved."

    sources = [
        metadata["source"].replace("\\", "/").split("/")[-1]
        for metadata in results["metadatas"][0]
    ]

    assert "Statistics.pdf" in sources, (
        f"Expected Statistics.pdf in the results, got: {sources}"
    )
