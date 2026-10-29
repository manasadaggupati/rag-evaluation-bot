
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


def find_relevant_ranks(results, expected_source, expected_page=None):
    ranks = []

    for rank, item in enumerate(results, start=1):
        if item["source"] != expected_source:
            continue

        if (
            expected_page is not None
            and item["page"] != int(expected_page)
        ):
            continue

        ranks.append(rank)

    return ranks


def main():
    questions = json.loads(
        QUESTIONS_FILE.read_text(encoding="utf-8")
    )

    print("Loading PDFs and chunks...")
    documents = load_pdfs(
        str(ROOT / "data" / "documents")
    )
    chunks = chunk_documents(
        documents,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    print("Loading models...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    reranker = CrossEncoder(RERANKING_MODEL)

    client = chromadb.EphemeralClient()
    collection = client.create_collection(
        name="reranking_diagnostic"
    )

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

    failures = []

    for question_data in questions:
        question = question_data["question"]
        expected_source = question_data["expected_source"]
        expected_page = question_data.get("expected_page")

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
            {
                "text": text,
                "source": metadata["source"],
                "page": int(metadata["page"]),
                "dense_rank": rank,
            }
            for rank, (text, metadata) in enumerate(
                zip(
                    retrieved["documents"][0],
                    retrieved["metadatas"][0],
                ),
                start=1,
            )
        ]

        baseline_top = candidates[:FINAL_K]

        scores = reranker.predict(
            [
                (question, item["text"])
                for item in candidates
            ],
            show_progress_bar=False,
        )

        # Rank all candidates, not just the final top five.
        reranked_all = sorted(
            [
                {**item, "score": float(score)}
                for item, score in zip(candidates, scores)
            ],
            key=lambda item: item["score"],
            reverse=True,
        )

        reranked = reranked_all[:FINAL_K]

        # Print all candidates for the question being investigated.
        if question == (
            "What is the purpose of reranking in a RAG pipeline?"
        ):
            print(
                "\nAll candidates ranked by CrossEncoder "
                "for the reranking question:"
            )

            for rank, item in enumerate(
                reranked_all, start=1
            ):
                print(
                    f"{rank}. {item['source']} "
                    f"p.{item['page']} "
                    f"| dense rank={item['dense_rank']} "
                    f"| score={item['score']:.4f} "
                    f"| {item['text'][:150]!r}"
                )

        baseline_ranks = find_relevant_ranks(
            baseline_top,
            expected_source,
            expected_page,
        )

        reranked_ranks = find_relevant_ranks(
            reranked,
            expected_source,
            expected_page,
        )

        # Record cases where baseline succeeds but reranking fails.
        if baseline_ranks and not reranked_ranks:
            candidate_has_expected_label = any(
                item["source"] == expected_source
                and (
                    expected_page is None
                    or item["page"] == int(expected_page)
                )
                for item in candidates
            )

            failures.append(
                {
                    "question": question,
                    "expected_source": expected_source,
                    "expected_page": expected_page,
                    "baseline": baseline_top,
                    "reranked": reranked,
                    "candidate_has_expected_label": (
                        candidate_has_expected_label
                    ),
                }
            )

    print("\n========== RERANKING FAILURE DIAGNOSTIC ==========")
    print(
        "Questions where baseline succeeded but reranking failed:",
        len(failures),
    )

    for number, failure in enumerate(failures, start=1):
        print(f"\nFailure {number}: {failure['question']}")
        print(f"Expected source: {failure['expected_source']}")
        print(f"Expected page: {failure['expected_page']}")
        print(
            "Expected label existed in top-20 candidates:",
            failure["candidate_has_expected_label"],
        )

        print("\nBaseline top 5:")
        for rank, item in enumerate(
            failure["baseline"], start=1
        ):
            print(
                f"  {rank}. {item['source']} "
                f"p.{item['page']} "
                f"| {item['text'][:180]!r}"
            )

        print("\nReranked top 5:")
        for rank, item in enumerate(
            failure["reranked"], start=1
        ):
            print(
                f"  {rank}. {item['source']} "
                f"p.{item['page']} "
                f"| score={item['score']:.4f} "
                f"| {item['text'][:180]!r}"
            )

    print(
        "\nDiagnostic complete. "
        "No persistent ChromaDB data was changed."
    )


if __name__ == "__main__":
    main()
