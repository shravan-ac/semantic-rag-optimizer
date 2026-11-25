
import csv
import json
import re
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parents[2]

DATASET_PATH = BASE_DIR / "evaluation" / "eval_dataset.json"
RESULTS_PATH = BASE_DIR / "evaluation" / "full_optimization_results.json"

CSV_OUTPUT_PATH = (
    BASE_DIR / "evaluation" / "full_optimization_quality_results.csv"
)
SUMMARY_OUTPUT_PATH = (
    BASE_DIR / "evaluation" / "full_optimization_quality_summary.json"
)

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "being", "to", "of", "in", "on", "for", "and", "or", "by",
    "with", "as", "at", "from", "that", "this", "it", "they",
    "their", "them", "he", "she", "we", "you", "your", "i",
    "do", "does", "did", "can", "could", "may", "might", "must",
    "should", "would", "will", "what", "who", "when", "how",
    "many", "much", "per", "each"
}

REFUSAL_PHRASES = [
    "do not specify",
    "does not specify",
    "not specified",
    "do not contain",
    "does not contain",
    "not mentioned",
    "not available",
    "couldn't find",
    "no relevant information",
    "available documents do not",
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def tokenize(text):
    words = re.findall(r"\b[a-z0-9]+\b", str(text).lower())
    return {
        word for word in words
        if word not in STOPWORDS
    }


def calculate_metrics(expected, generated):
    expected_tokens = tokenize(expected)
    generated_tokens = tokenize(generated)

    if not expected_tokens or not generated_tokens:
        return 0.0, 0.0, 0.0

    common = expected_tokens.intersection(generated_tokens)

    precision = len(common) / len(generated_tokens)
    recall = len(common) / len(expected_tokens)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def detect_refusal(answer):
    answer = str(answer).lower()
    return any(phrase in answer for phrase in REFUSAL_PHRASES)


def main():
    dataset = load_json(DATASET_PATH)
    results = load_json(RESULTS_PATH)

    if isinstance(results, dict):
        results = results.get("results", [])

    results_by_question = {
        item.get("question", "").strip(): item
        for item in results
    }

    rows = []

    for item in dataset:
        question = item["question"].strip()
        result = results_by_question.get(question)

        if result is None:
            rows.append({
                "question": question,
                "answerable": item["answerable"],
                "generated_answer": "",
                "expected_answer": item.get("expected_answer", ""),
                "cache_hit": "",
                "model": "",
                "evaluation_type": "MISSING_RESULT",
                "precision": "",
                "recall": "",
                "token_f1": "",
                "refusal_detected": "",
                "assessment": "MISSING_RESULT",
            })
            continue

        generated_answer = (
            result.get("answer")
            or result.get("generated_answer")
            or ""
        )

        expected_answer = item.get("expected_answer", "")
        answerable = item["answerable"]
        cache_hit = result.get("cache_hit", False)
        model = result.get("model", "")

        if result.get("error") or not generated_answer:
            evaluation_type = "ERROR"
            precision = recall = f1 = ""
            refusal_detected = ""
            assessment = "GENERATION_ERROR"

        elif answerable:
            precision, recall, f1 = calculate_metrics(
                expected_answer,
                generated_answer
            )
            refusal_detected = detect_refusal(generated_answer)
            evaluation_type = "ANSWERABLE"
            assessment = "MANUAL_REVIEW_REQUIRED"

        else:
            precision = recall = f1 = ""
            refusal_detected = detect_refusal(generated_answer)
            evaluation_type = "UNANSWERABLE"
            assessment = (
                "CORRECT_REFUSAL"
                if refusal_detected
                else "CHECK_REFUSAL"
            )

        rows.append({
            "question": question,
            "answerable": answerable,
            "generated_answer": generated_answer,
            "expected_answer": expected_answer,
            "cache_hit": cache_hit,
            "model": model,
            "evaluation_type": evaluation_type,
            "precision": precision,
            "recall": recall,
            "token_f1": f1,
            "refusal_detected": refusal_detected,
            "assessment": assessment,
        })

    # Calculate summary metrics.
    answerable_rows = [
        row for row in rows
        if row["evaluation_type"] == "ANSWERABLE"
    ]

    scored_rows = [
        row for row in answerable_rows
        if row["token_f1"] != ""
    ]

    unanswerable_rows = [
        row for row in rows
        if row["evaluation_type"] == "UNANSWERABLE"
    ]

    correct_refusals = sum(
        row["refusal_detected"] is True
        for row in unanswerable_rows
    )

    cache_hits = [
        row for row in rows
        if row["cache_hit"] is True
    ]

    cache_misses = [
        row for row in rows
        if row["cache_hit"] is False
    ]

    def average_f1(group):
        values = [
            row["token_f1"]
            for row in group
            if isinstance(row["token_f1"], (int, float))
        ]
        return round(sum(values) / len(values), 4) if values else None

    summary = {
        "total_dataset_questions": len(dataset),
        "evaluated_questions": len(rows),
        "missing_results": sum(
            row["evaluation_type"] == "MISSING_RESULT"
            for row in rows
        ),
        "generation_errors": sum(
            row["evaluation_type"] == "ERROR"
            for row in rows
        ),
        "answerable_questions_scored": len(scored_rows),
        "average_answerable_token_f1": average_f1(scored_rows),
        "unanswerable_questions_checked": len(unanswerable_rows),
        "correct_refusal_detections": correct_refusals,
        "refusal_detection_rate": (
            round(correct_refusals / len(unanswerable_rows), 4)
            if unanswerable_rows else None
        ),
        "cache_hits": len(cache_hits),
        "cache_misses": len(cache_misses),
        "cache_hit_answerable_token_f1": average_f1([
            row for row in scored_rows if row["cache_hit"] is True
        ]),
        "cache_miss_answerable_token_f1": average_f1([
            row for row in scored_rows if row["cache_hit"] is False
        ]),
        "metric_note": (
            "Token F1 measures lexical overlap, not semantic or factual "
            "correctness. Manually review answerable outputs. Refusal "
            "detection uses phrase matching and may produce false positives."
        ),
    }

    # Save detailed CSV.
    with open(CSV_OUTPUT_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)

    # Save summary JSON.
    with open(SUMMARY_OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\nFULL OPTIMIZATION ANSWER QUALITY")
    print("=" * 45)

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nDetailed results:", CSV_OUTPUT_PATH)
    print("Summary:", SUMMARY_OUTPUT_PATH)


if __name__ == "__main__":
    main()
