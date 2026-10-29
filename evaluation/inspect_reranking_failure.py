
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.pdf_loader import load_pdfs
from src.chunking import chunk_documents


ROOT = Path(__file__).resolve().parent.parent

QUESTION = (
    "How does a cross-encoder reranker score retrieved passages "
    "and select the most relevant passages for the final context "
    "in a RAG pipeline?"
)
EXPECTED_SOURCE = "RAG.pdf"
EXPECTED_PAGE = 10

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKING_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 0
CANDIDATE_K = 20
FINAL_K = 5


def main():
    print("Loading PDFs and creating chunks...")
    documents = load_pdfs(str(ROOT / "data" / "documents"))
    chunks = chunk_documents(
        documents,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    print("Loading models...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    reranker = CrossEncoder(RERANKING_MODEL)

    client = chromadb.EphemeralClient()
    collection = client.create_collection(name="inspect_reranking_failure")

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedder.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[
            {
                "source": chunk["source"],
                "page": int(chunk["page"]),
            }
            for chunk in chunks
        ],
    )

    query_embedding = embedder.encode(
        [QUESTION],
        normalize_embeddings=True,
    )[0]

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(CANDIDATE_K, len(chunks)),
        include=["documents", "metadatas", "distances"],
    )

    candidates = []
    for rank, (text, metadata, distance) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ),
        start=1,
    ):
        candidates.append({
            "text": text,
            "source": metadata["source"],
            "page": int(metadata["page"]),
            "dense_rank": rank,
            "distance": float(distance),
        })

    scores = reranker.predict(
        [(QUESTION, item["text"]) for item in candidates],
        show_progress_bar=False,
    )

    for item, score in zip(candidates, scores):
        item["rerank_score"] = float(score)

    reranked = sorted(
        candidates,
        key=lambda item: item["rerank_score"],
        reverse=True,
    )

    print("\n========== BASELINE TOP 10 ==========")
    for item in candidates[:10]:
        marker = (
            " <-- EXPECTED PAGE"
            if item["source"] == EXPECTED_SOURCE
            and item["page"] == EXPECTED_PAGE
            else ""
        )
        print(
            f"\nDense rank {item['dense_rank']} | "
            f"{item['source']} p.{item['page']} | "
            f"distance={item['distance']:.4f}{marker}"
        )
        print(item["text"][:350].replace("\n", " "))

    print("\n========== RERANKED TOP 10 ==========")
    for rank, item in enumerate(reranked[:10], start=1):
        marker = (
            " <-- EXPECTED PAGE"
            if item["source"] == EXPECTED_SOURCE
            and item["page"] == EXPECTED_PAGE
            else ""
        )
        print(
            f"\nRerank {rank} | original dense rank "
            f"{item['dense_rank']} | "
            f"{item['source']} p.{item['page']} | "
            f"score={item['rerank_score']:.4f}{marker}"
        )
        print(item["text"][:350].replace("\n", " "))

    expected_candidates = [
        item for item in candidates
        if item["source"] == EXPECTED_SOURCE
        and item["page"] == EXPECTED_PAGE
    ]

    print("\n========== SUMMARY ==========")
    print(f"Top-20 candidates: {len(candidates)}")
    print(f"Expected-page chunks in candidates: {len(expected_candidates)}")

    for item in expected_candidates:
        rerank_position = reranked.index(item) + 1
        print(
            f"Expected page chunk: dense rank={item['dense_rank']}, "
            f"rerank position={rerank_position}, "
            f"score={item['rerank_score']:.4f}"
        )

    print("\nInspection complete. Persistent ChromaDB was not modified.")


if __name__ == "__main__":
    main()
