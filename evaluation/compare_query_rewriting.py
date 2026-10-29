
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.pdf_loader import load_pdfs
from src.chunking import chunk_documents


ROOT = Path(__file__).resolve().parent.parent
LABELS_FILE = ROOT / "evaluation" / "relevance_labels.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKING_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 0
CANDIDATE_K = 20
FINAL_K = 5

# These rewrites are hypotheses to test, not guaranteed improvements.
REWRITES = {
    "What is an embedding in machine learning?":
        "How do embedding models represent text as numerical vectors "
        "for semantic similarity search?",

    "What is the purpose of reranking in a RAG pipeline?":
        "How does a cross-encoder reranker score retrieved passages "
        "and select the most relevant passages for the final context "
        "in a RAG pipeline?",

    "What is the role of embeddings in machine learning?":
        "How are numerical embeddings used to represent data and "
        "capture semantic relationships in machine learning?",

    "What is the purpose of tokenization in language models?":
        "How does a tokenizer convert text into token IDs, and why "
        "does tokenization affect context length and language model input?",
}


def reciprocal_rank(results, relevant_pages):
    for rank, result in enumerate(results, start=1):
        if (result["source"], result["page"]) in relevant_pages:
            return 1.0 / rank
    return 0.0


def main():
    labels = json.loads(LABELS_FILE.read_text(encoding="utf-8"))

    print("Loading PDFs and chunks...")
    documents = load_pdfs(str(ROOT / "data" / "documents"))
    chunks = chunk_documents(
        documents,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    print("Loading embedding and reranking models...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    reranker = CrossEncoder(RERANKING_MODEL)

    # Temporary in-memory collection; persistent ChromaDB is untouched.
    client = chromadb.EphemeralClient()
    collection = client.create_collection(name="query_rewrite_comparison")

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
            {"source": chunk["source"], "page": int(chunk["page"])}
            for chunk in chunks
        ],
    )

    # Four conditions:
    # Original query + dense retrieval
    # Original query + reranking
    # Rewritten query + dense retrieval
    # Rewritten query + reranking
    conditions = {
        "Original / Dense": [],
        "Original / Reranked": [],
        "Rewritten / Dense": [],
        "Rewritten / Reranked": [],
    }

    for index, item in enumerate(labels, start=1):
        original_query = item["question"]
        rewritten_query = REWRITES[original_query]

        relevant_pages = {
            (p["source"], int(p["page"]))
            for p in item["relevant_passages"]
        }

        print(f"\nQ{index}: {original_query}")
        print(f"Rewrite: {rewritten_query}")

        for query_type, query in [
            ("Original", original_query),
            ("Rewritten", rewritten_query),
        ]:
            query_embedding = embedder.encode(
                [query],
                normalize_embeddings=True,
            )[0]

            retrieved = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(CANDIDATE_K, len(chunks)),
                include=["documents", "metadatas"],
            )

            candidates = [
                {
                    "text": text,
                    "source": metadata["source"],
                    "page": int(metadata["page"]),
                }
                for text, metadata in zip(
                    retrieved["documents"][0],
                    retrieved["metadatas"][0],
                )
            ]

            dense_top = candidates[:FINAL_K]

            scores = reranker.predict(
                [(query, candidate["text"]) for candidate in candidates],
                show_progress_bar=False,
            )

            reranked_top = [
                candidate
                for candidate, score in sorted(
                    zip(candidates, scores),
                    key=lambda pair: float(pair[1]),
                    reverse=True,
                )[:FINAL_K]
            ]

            dense_rr = reciprocal_rank(dense_top, relevant_pages)
            reranked_rr = reciprocal_rank(reranked_top, relevant_pages)

            conditions[f"{query_type} / Dense"].append(dense_rr)
            conditions[f"{query_type} / Reranked"].append(reranked_rr)

            print(
                f"  {query_type:9} | "
                f"Dense RR={dense_rr:.3f} | "
                f"Reranked RR={reranked_rr:.3f}"
            )

    print("\n========== QUERY REWRITING COMPARISON ==========")
    print("Metric: MRR@5 using manually labelled source/page pairs")
    print(f"Questions: {len(labels)}")

    for name, values in conditions.items():
        print(f"{name:22}: MRR@5 = {sum(values) / len(values):.3f}")

    print(
        "\nCaution: only four questions are labelled. "
        "Page matches are proxies for passage relevance."
    )
    print("Persistent ChromaDB was not modified.")


if __name__ == "__main__":
    main()
