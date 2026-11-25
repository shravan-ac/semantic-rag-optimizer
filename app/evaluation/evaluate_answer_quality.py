
import csv
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATASET_PATH = BASE_DIR / "evaluation" / "eval_dataset.json"
RESULTS_PATH = BASE_DIR / "evaluation" / "generation_only_results.json"

CSV_PATH = BASE_DIR / "evaluation" / "answer_quality_results.csv"
SUMMARY_PATH = BASE_DIR / "evaluation" / "answer_quality_summary.json"

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be",
    "to", "of", "in", "on", "for", "and", "or", "that",
    "this", "these", "those", "it", "they", "their",
    "do", "does", "can", "could", "may", "must",
    "employees", "employee", "based", "provided"
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
    "no relevant information"
]


def tokenize(text):
    words = re.findall(r"\b[a-z0-9]+\b", text.lower())
    return [word for word in words if word not in STOPWORDS]


def calculate_token_scores(reference, generated):
    reference_tokens = set(tokenize(reference))
    generated_tokens = set(tokenize(generated))

    if not reference_tokens:
        return 0.0, 0.0, 0.0

    overlap = reference_tokens & generated_tokens

    precision = (
        len(overlap) / len(generated_tokens)
        if generated_tokens else 0.0
    )

    recall = len(overlap) / len(reference_tokens)

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return precision, recall, f1


def is_refusal(answer):
    answer = answer.lower()
    return any(phrase in answer for phrase in REFUSAL_PHRASES)


def main():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    with open(RESULTS_PATH, "r", encoding="utf-8") as file:
        results_data = json.load(file)

    # Support either a list or a dictionary containing results.
    results = (
        results_data
        if isinstance(results_data, list)
        else results_data.get("results", [])
    )

    results_by_question = {
        item["question"]: item
        for item in results
        if item.get("question")
    }

    rows = []

    answerable_scores = []
    refusal_correct = 0
    refusal_total = 0
    missing_results = 0

    for item in dataset:
        question = item["question"]
        reference = item.get("expected_answer", "")
        answerable = item.get("answerable", True)

        generated_item = results_by_question.get(question)

        if not generated_item:
            missing_results += 1
            continue

        answer = generated_item.get("answer", "")
        status = generated_item.get("status", "SUCCESS")

        if status != "SUCCESS" or not answer:
            continue

        refusal = is_refusal(answer)

        if answerable:
            precision, recall, f1 = calculate_token_scores(
                reference, answer
            )
            answerable_scores.append(f1)
            quality_score = round(f1, 4)
            evaluation_type = "REFERENCE_TOKEN_F1"
            assessment = "MANUAL_REVIEW_REQUIRED"

        else:
            precision = None
            recall = None
            f1 = None
            quality_score = None
            evaluation_type = "REFUSAL_CHECK"

            refusal_total += 1

            if refusal:
                refusal_correct += 1
                assessment = "REFUSAL_DETECTED"
            else:
                assessment = "POSSIBLE_UNSUPPORTED_ANSWER"

        rows.append({
            "question": question,
            "answerable": answerable,
            "generated_answer": answer,
            "expected_answer": reference,
            "evaluation_type": evaluation_type,
            "precision": precision,
            "recall": recall,
            "token_f1": f1,
            "quality_score": quality_score,
            "refusal_detected": refusal,
            "assessment": assessment
        })

    fieldnames = list(rows[0].keys()) if rows else []

    with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)

    average_f1 = (
        sum(answerable_scores) / len(answerable_scores)
        if answerable_scores else 0.0
    )

    refusal_accuracy = (
        refusal_correct / refusal_total
        if refusal_total else 0.0
    )

    summary = {
        "total_dataset_questions": len(dataset),
        "evaluated_questions": len(rows),
        "missing_results": missing_results,
        "answerable_questions_scored": len(answerable_scores),
        "average_answerable_token_f1": round(average_f1, 4),
        "unanswerable_questions_checked": refusal_total,
        "correct_refusal_detections": refusal_correct,
        "refusal_detection_rate": round(refusal_accuracy, 4),
        "metric_note": (
            "Token F1 measures lexical overlap, not semantic correctness. "
            "Manual review is required for factual accuracy and completeness. "
            "Refusal detection uses phrase matching."
        )
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\nANSWER QUALITY EVALUATION")
    print("=" * 45)
    print(f"Evaluated questions: {len(rows)}")
    print(f"Missing results: {missing_results}")
    print(f"Average answerable Token F1: {average_f1:.4f}")
    print(f"Correct refusal detections: {refusal_correct}/{refusal_total}")
    print(f"Refusal detection rate: {refusal_accuracy:.2%}")
    print(f"\nCSV saved: {CSV_PATH}")
    print(f"Summary saved: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
