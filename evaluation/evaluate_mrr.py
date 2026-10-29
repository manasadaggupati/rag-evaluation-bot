
import json
from pathlib import Path

from src.vector_store import search_chunks

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_FILE = PROJECT_ROOT / "evaluation" / "test_questions.json"


def evaluate_mrr():
    with open(TEST_FILE, "r", encoding="utf-8") as file:
        questions = json.load(file)

    reciprocal_ranks = []

    for item in questions:
        results = search_chunks(item["question"], top_k=5)
        metadatas = results["metadatas"][0]
        expected = item["expected_source"]

        rank = next(
            (
                index
                for index, metadata in enumerate(metadatas, start=1)
                if Path(metadata["source"]).name == expected
            ),
            None,
        )

        reciprocal_rank = 1 / rank if rank else 0
        reciprocal_ranks.append(reciprocal_rank)

        print(f"\nQuestion: {item['question']}")
        print(f"Expected source: {expected}")
        print(f"Rank: {rank if rank else 'Not found'}")
        print(f"Reciprocal rank: {reciprocal_rank:.2f}")

    mrr = (
        sum(reciprocal_ranks) / len(reciprocal_ranks)
        if reciprocal_ranks
        else 0
    )

    print("\n" + "=" * 40)
    print(f"Mean Reciprocal Rank (MRR): {mrr:.3f}")
    print("=" * 40)


if __name__ == "__main__":
    evaluate_mrr()
