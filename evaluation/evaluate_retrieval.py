
from src.vector_store import search_chunks
import json
from pathlib import Path

QUESTIONS_FILE = Path("evaluation/test_questions.json")
TOP_K = 5


def main():
    with QUESTIONS_FILE.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    page_hits = 0
    source_hits = 0

    for item in questions:
        question = item["question"]
        expected_source = item["expected_source"]
        expected_page = item.get("expected_page")

        results = search_chunks(question, top_k=TOP_K)
        metadatas = results.get("metadatas", [[]])[0]

        retrieved_sources = [
            metadata["source"] for metadata in metadatas
        ]

        source_hit = expected_source in retrieved_sources
        page_hit = any(
            metadata["source"] == expected_source
            and metadata["page"] == expected_page
            for metadata in metadatas
        ) if expected_page is not None else False

        source_hits += int(source_hit)
        page_hits += int(page_hit)

        print(f"\nQuestion: {question}")
        print(f"Expected: {expected_source}, page {expected_page}")
        print(f"Retrieved: {[(m['source'], m['page']) for m in metadatas]}")
        print(f"Source hit: {'YES' if source_hit else 'NO'}")
        if expected_page is not None:
            print(f"Page hit: {'YES' if page_hit else 'NO'}")

    total = len(questions)

    print("\n" + "=" * 40)
    print(f"Source Hit Rate@{TOP_K}: {source_hits / total:.2%}")
    page_labelled = sum(
        1 for item in questions if item.get("expected_page") is not None
    )
    if page_labelled:
        print(
            f"Expected Page Hit Rate@{TOP_K}: "
            f"{page_hits / page_labelled:.2%} "
            f"({page_hits}/{page_labelled})"
        )
    print("=" * 40)


if __name__ == "__main__":
    main()
