
import csv
import json
import os
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from google import genai

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
TEST_FILE = EVALUATION_DIR / "test_answers.json"
CSV_FILE = EVALUATION_DIR / "answer_evaluation.csv"

load_dotenv(PROJECT_ROOT / ".env")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = "gemini-3.5-flash-lite"


def get_chat_answer(question):
    payload = json.dumps({"question": question}).encode("utf-8")
    request = urllib.request.Request(
        "http://127.0.0.1:8000/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=90) as response:
        data = json.loads(response.read().decode("utf-8"))

    return data["answer"]


def score_answer(question, expected, generated):
    prompt = f"""
Evaluate the generated answer against the reference answer.

Question: {question}

Reference answer:
{expected}

Generated answer:
{generated}

Score correctness and completeness:
5 = Correct and complete
4 = Mostly correct
3 = Partially correct
2 = Mostly incorrect
1 = Incorrect or irrelevant

Return only a single integer from 1 to 5.
"""
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
    )

    score_text = response.text.strip()
    score = int(score_text)

    if score not in range(1, 6):
        raise ValueError(f"Invalid judge score: {score_text}")

    return score


def evaluate_answers():
    with open(TEST_FILE, "r", encoding="utf-8") as file:
        test_cases = json.load(file)

    rows = []

    for item in test_cases:
        question = item["question"]
        expected = item["expected_answer"]
        start = time.perf_counter()

        try:
            generated = get_chat_answer(question)
            latency_seconds = round(time.perf_counter() - start, 2)
            score = score_answer(question, expected, generated)

            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "question": question,
                "answer_quality_score": score,
                "latency_seconds": latency_seconds,
                "status": "success",
            }

            print(f"\nQuestion: {question}")
            print(f"Answer quality: {score}/5")
            print(f"Latency: {latency_seconds:.2f} seconds")

        except Exception as error:
            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "question": question,
                "answer_quality_score": "",
                "latency_seconds": round(time.perf_counter() - start, 2),
                "status": f"failed: {type(error).__name__}",
            }
            print(f"\nFailed: {question}")
            print(f"{type(error).__name__}: {error}")

        rows.append(row)

    fields = [
        "timestamp",
        "question",
        "answer_quality_score",
        "latency_seconds",
        "status",
    ]

    with open(CSV_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    successful = [r for r in rows if r["status"] == "success"]

    if successful:
        average_score = sum(
            int(r["answer_quality_score"]) for r in successful
        ) / len(successful)

        average_latency = sum(
            float(r["latency_seconds"]) for r in successful
        ) / len(successful)

        print("\n" + "=" * 40)
        print(f"Average answer quality: {average_score:.2f}/5")
        print(f"Average latency: {average_latency:.2f} seconds")
        print(f"Successful evaluations: {len(successful)}/{len(rows)}")
        print(f"CSV saved to: {CSV_FILE}")
        print("=" * 40)


if __name__ == "__main__":
    evaluate_answers()
