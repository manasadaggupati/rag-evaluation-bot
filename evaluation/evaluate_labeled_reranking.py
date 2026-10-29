
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


def reciprocal_rank(results, relevant_pages):
    for rank, item in enumerate(results, start=1):
        if (item["source"], item["page"]) in relevant_pages:
            return 1 / rank
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

    print("Loading models...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    reranker = CrossEncoder(RERANKING_MODEL)

    client = chromadb.EphemeralClient()
    collection = client.create_collection(name="labeled_reranking_eval")

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

    baseline_rr = []
    reranked_rr = []

    for index, item in enumerate(labels, start=1):
        question = item["question"]
        relevant_pages = {
            (passage["source"], int(passage["page"]))
            for passage in item["relevant_passages"]
        }

        query_embedding = embedder.encode(
            [question],
            normalize_embeddings=True,
        )[0]

        retrieved = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(CANDIDATE_K, len(chunks)),
            include=["documents", "metadatas"],
        )

        candidates = [
            {"text": text, "source": metadata["source"],
             "page": int(metadata["page"])}
            for text, metadata in zip(
                retrieved["documents"][0],
                retrieved["metadatas"][0],
            )
        ]

        baseline_top = candidates[:FINAL_K]

        scores = reranker.predict(
            [(question, candidate["text"]) for candidate in candidates],
            show_progress_bar=False,
        )

        reranked_top = [
            candidate
            for candidate, score in sorted(
                zip(
                    [
                        {**candidate, "score": float(score)}
                        for candidate, score in zip(candidates, scores)
                    ],
                    scores,
                ),
                key=lambda pair: pair[1],
                reverse=True,
            )[:FINAL_K]
        ]

        baseline_score = reciprocal_rank(baseline_top, relevant_pages)
        reranked_score = reciprocal_rank(reranked_top, relevant_pages)

        baseline_rr.append(baseline_score)
        reranked_rr.append(reranked_score)

        print(f"\nQ{index}: {question}")
        print(f"  Baseline relevant-page RR: {baseline_score:.3f}")
        print(f"  Reranked relevant-page RR: {reranked_score:.3f}")

    print("\n========== MANUALLY LABELED RESULTS ==========")
    print(f"Questions: {len(labels)}")
    print(f"Baseline MRR@{FINAL_K}: {sum(baseline_rr) / len(baseline_rr):.3f}")
    print(f"Reranked MRR@{FINAL_K}: {sum(reranked_rr) / len(reranked_rr):.3f}")
    print("Interpret results cautiously: this is a small, manually labeled set.")


if __name__ == "__main__":
    main()
