
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.pdf_loader import load_pdfs
from src.chunking import chunk_documents


ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_FILE = ROOT / "evaluation" / "test_questions.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKING_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 0
CANDIDATE_K = 20
FINAL_K = 5


def reciprocal_rank(relevant_ranks):
    if not relevant_ranks:
        return 0.0
    return 1.0 / min(relevant_ranks)


def main():
    print("Loading benchmark questions...")
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))

    print("Loading PDFs and creating chunks...")
    documents = load_pdfs(str(ROOT / "data" / "documents"))
    chunks = chunk_documents(
        documents,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    print(f"Total chunks: {len(chunks)}")

    print("Loading embedding model...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    print("Loading CrossEncoder reranking model...")
    reranker = CrossEncoder(RERANKING_MODEL)

    # EphemeralClient keeps this experiment separate from your saved database.
    client = chromadb.EphemeralClient()
    collection = client.create_collection(name="reranking_experiment")

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

    baseline_source_hits = 0
    reranked_source_hits = 0
    baseline_source_rr = []
    reranked_source_rr = []

    page_questions = 0
    baseline_page_hits = 0
    reranked_page_hits = 0
    baseline_page_rr = []
    reranked_page_rr = []

    for index, question_data in enumerate(questions, start=1):
        question = question_data["question"]
        expected_source = question_data["expected_source"]
        expected_page = question_data.get("expected_page")

        query_embedding = embedder.encode(
            [question],
            normalize_embeddings=True,
        )[0]

        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(CANDIDATE_K, len(chunks)),
            include=["documents", "metadatas", "distances"],
        )

        candidates = []
        for i, (text, metadata) in enumerate(
            zip(results["documents"][0], results["metadatas"][0])
        ):
            candidates.append({
                "text": text,
                "source": metadata["source"],
                "page": int(metadata["page"]),
                "dense_rank": i + 1,
            })

        # Baseline: the first five chunks from vector search.
        baseline_top = candidates[:FINAL_K]

        # Rerank the same candidate pool with query-passage relevance scores.
        pairs = [(question, item["text"]) for item in candidates]
        scores = reranker.predict(pairs, show_progress_bar=False)

        reranked = sorted(
            [
                {**item, "rerank_score": float(score)}
                for item, score in zip(candidates, scores)
            ],
            key=lambda item: item["rerank_score"],
            reverse=True,
        )
        reranked_top = reranked[:FINAL_K]

        baseline_source_ranks = [
            rank for rank, item in enumerate(baseline_top, start=1)
            if item["source"] == expected_source
        ]
        reranked_source_ranks = [
            rank for rank, item in enumerate(reranked_top, start=1)
            if item["source"] == expected_source
        ]

        baseline_source_hits += bool(baseline_source_ranks)
        reranked_source_hits += bool(reranked_source_ranks)
        baseline_source_rr.append(reciprocal_rank(baseline_source_ranks))
        reranked_source_rr.append(reciprocal_rank(reranked_source_ranks))

        if expected_page is not None:
            page_questions += 1

            baseline_page_ranks = [
                rank for rank, item in enumerate(baseline_top, start=1)
                if item["source"] == expected_source
                and item["page"] == int(expected_page)
            ]
            reranked_page_ranks = [
                rank for rank, item in enumerate(reranked_top, start=1)
                if item["source"] == expected_source
                and item["page"] == int(expected_page)
            ]

            baseline_page_hits += bool(baseline_page_ranks)
            reranked_page_hits += bool(reranked_page_ranks)
            baseline_page_rr.append(reciprocal_rank(baseline_page_ranks))
            reranked_page_rr.append(reciprocal_rank(reranked_page_ranks))

        print(f"\nQ{index}: {question}")
        print(
            "  Baseline sources:",
            [f'{x["source"]} p.{x["page"]}' for x in baseline_top],
        )
        print(
            "  Reranked sources:",
            [f'{x["source"]} p.{x["page"]}' for x in reranked_top],
        )

    total = len(questions)

    print("\n========== RERANKING RESULTS ==========")
    print(f"Questions: {total}")
    print(f"Candidate depth: {CANDIDATE_K}; final top-k: {FINAL_K}")

    print("\nSource retrieval:")
    print(
        f"Baseline Hit Rate@{FINAL_K}: "
        f"{baseline_source_hits / total:.3f}"
    )
    print(
        f"Reranked Hit Rate@{FINAL_K}: "
        f"{reranked_source_hits / total:.3f}"
    )
    print(
        f"Baseline Source MRR@{FINAL_K}: "
        f"{sum(baseline_source_rr) / total:.3f}"
    )
    print(
        f"Reranked Source MRR@{FINAL_K}: "
        f"{sum(reranked_source_rr) / total:.3f}"
    )

    if page_questions:
        print(f"\nExpected-page retrieval ({page_questions} questions):")
        print(
            f"Baseline Page Hit Rate@{FINAL_K}: "
            f"{baseline_page_hits / page_questions:.3f}"
        )
        print(
            f"Reranked Page Hit Rate@{FINAL_K}: "
            f"{reranked_page_hits / page_questions:.3f}"
        )
        print(
            f"Baseline Page MRR@{FINAL_K}: "
            f"{sum(baseline_page_rr) / page_questions:.3f}"
        )
        print(
            f"Reranked Page MRR@{FINAL_K}: "
            f"{sum(reranked_page_rr) / page_questions:.3f}"
        )

    print("\nExperiment complete. Your persistent ChromaDB was not modified.")


if __name__ == "__main__":
    main()
