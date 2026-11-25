
import json
import uuid
from pathlib import Path

from qdrant_client import QdrantClient

from app.cache import semantic_cache


BASE_DIR = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    BASE_DIR
    / "app"
    / "evaluation"
    / "cache_pairs_holdout.json"
)

OUTPUT_PATH = (
    BASE_DIR
    / "evaluation"
    / "cache_holdout_isolated_results.json"
)

QDRANT_URL = "http://localhost:6333"


def evaluate_holdout():

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        pairs = json.load(file)

    qdrant = QdrantClient(url=QDRANT_URL)

    results = []

    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    print("\nISOLATED SEMANTIC CACHE HOLDOUT EVALUATION")
    print("=" * 55)
    print("Total pairs:", len(pairs))
    print("Equivalent pairs:", sum(p["equivalent"] for p in pairs))
    print("Non-equivalent pairs:", sum(not p["equivalent"] for p in pairs))

    for index, pair in enumerate(pairs, start=1):

        # A unique collection ensures each pair is evaluated independently.
        collection_name = f"cache_holdout_{uuid.uuid4().hex[:12]}"

        question_a = pair["question_a"]
        question_b = pair["question_b"]
        expected_hit = pair["equivalent"]

        print(f"\nPair {index}/{len(pairs)} | ID: {pair['id']}")
        print("Question A:", question_a)
        print("Question B:", question_b)

        try:
            semantic_cache.create_cache_collection(
                collection_name=collection_name
            )

            semantic_cache.save_to_cache(
                query=question_a,
                answer="Isolated cache validation answer.",
                sources=["leave_policy.txt"],
                collection_name=collection_name
            )

            result = semantic_cache.search_cache(
                question_b,
                collection_name=collection_name
            )

            actual_hit = result is not None

            if expected_hit and actual_hit:
                true_positive += 1
                outcome = "TRUE_POSITIVE"

            elif not expected_hit and actual_hit:
                false_positive += 1
                outcome = "FALSE_POSITIVE"

            elif not expected_hit and not actual_hit:
                true_negative += 1
                outcome = "TRUE_NEGATIVE"

            else:
                false_negative += 1
                outcome = "FALSE_NEGATIVE"

            row = {
                "id": pair["id"],
                "category": pair["category"],
                "question_a": question_a,
                "question_b": question_b,
                "expected_equivalent": expected_hit,
                "actual_cache_hit": actual_hit,
                "outcome": outcome,
                "cache_score": (
                    result.get("score") if result else None
                ),
            }

            results.append(row)

            print("Expected:", "HIT" if expected_hit else "MISS")
            print("Actual:", "HIT" if actual_hit else "MISS")
            print("Outcome:", outcome)

        except Exception as error:

            results.append({
                "id": pair["id"],
                "category": pair["category"],
                "question_a": question_a,
                "question_b": question_b,
                "expected_equivalent": expected_hit,
                "actual_cache_hit": None,
                "outcome": "ERROR",
                "error": str(error),
            })

            print("Evaluation error:", error)

        finally:
            # Delete only the uniquely named collection created for this pair.
            if qdrant.collection_exists(collection_name):
                qdrant.delete_collection(collection_name)

    evaluated = true_positive + false_positive + true_negative + false_negative

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive else 0.0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    summary = {
        "total_pairs": len(pairs),
        "evaluated_pairs": evaluated,
        "errors": len(pairs) - evaluated,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }

    output = {
        "summary": summary,
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4, ensure_ascii=False)

    print("\n" + "=" * 55)
    print("HOLDOUT EVALUATION SUMMARY")
    print("=" * 55)

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nResults saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    evaluate_holdout()
