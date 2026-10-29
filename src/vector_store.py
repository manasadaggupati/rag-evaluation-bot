
import chromadb
from src.embeddings import create_embeddings

client = chromadb.PersistentClient(path="./chroma_db")

COLLECTION_NAME = "rag_documents"


def get_collection():
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


collection = get_collection()


def index_chunks(chunks: list[dict], reset: bool = False):
    global collection

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        collection = get_collection()

    if not chunks:
        return 0

    texts = [chunk["text"] for chunk in chunks]
    embeddings = create_embeddings(texts)

    collection.upsert(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[
            {
                "source": chunk["source"],
                "page": chunk["page"],
                "start_index": chunk["start_index"],
            }
            for chunk in chunks
        ],
    )

    return len(chunks)


def search_chunks(question: str, top_k: int = 5):
    query_embedding = create_embeddings([question])[0]

    return collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
