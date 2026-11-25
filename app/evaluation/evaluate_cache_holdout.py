
import json
from pathlib import Path

import numpy as np

from app.cache.semantic_validator import validate_question_pairs


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

HOLDOUT_FILE = (
    BASE_DIR
    / "app"
    / "evaluation"
    / "cache_pairs_holdout.json"
)

# Frozen thresholds selected using the development dataset.
MEAN_THRESHOLD = 0.27458
MINIMUM_THRESHOLD = 0.00224


# --------------------------------------------------
# Load holdout dataset
# --------------------------------------------------

def load_dataset():
    with open(HOLDOUT_FILE, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not dataset:
        raise ValueError("Holdout dataset is empty.")

    required_fields = [
        "id",
        "question_a",
        "question_b",
        "equivalent",
    ]

    for index, item in enumerate(dataset):
        for field in required_fields:
            if field not in item:
                raise ValueError(
                    f"Missing field '{field}' "
                    f"in holdout pair at index {index}."
                )

    return dataset


# --------------------------------------------------
# Calculate classification metrics
# --------------------------------------------------

def calculate_metrics(labels, predictions):
    tp = sum(
        actual == 1 and predicted == 1
        for actual, predicted in zip(labels, predictions)
    )

    fp = sum(
        actual == 0 and predicted == 1
        for actual, predicted in zip(labels, predictions)
    )

    tn = sum(
        actual == 0 and predicted == 0
        for actual, predicted in zip(labels, predictions)
    )

    fn = sum(
        actual == 1 and predicted == 0
        for actual, predicted in zip(labels, predictions)
    )

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# --------------------------------------------------
# Evaluate one validation strategy
# --------------------------------------------------

def evaluate_strategy(name, scores, threshold, labels):
    predictions = [
        1 if score >= threshold else 0
        for score in scores
    ]

    metrics = calculate_metrics(labels, predictions)

    print(f"\n{'=' * 55}")
    print(name)
    print(f"{'=' * 55}")

    print(f"Threshold: {threshold:.5f}")
    print(f"TP: {metrics['TP']}")
    print(f"FP: {metrics['FP']}")
    print(f"TN: {metrics['TN']}")
    print(f"FN: {metrics['FN']}")

    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1 Score:  {metrics['f1']:.3f}")

    return predictions, metrics


# --------------------------------------------------
# Main evaluation
# --------------------------------------------------

def main():
    dataset = load_dataset()

    print(f"Loaded {len(dataset)} holdout question pairs.")

    questions_a = [item["question_a"] for item in dataset]
    questions_b = [item["question_b"] for item in dataset]

    labels = [
        1 if item["equivalent"] else 0
        for item in dataset
    ]

    # NLI is directional, so evaluate both directions.
    pairs_forward = list(zip(questions_a, questions_b))
    pairs_reverse = list(zip(questions_b, questions_a))

    print("\nScoring question pairs: forward direction...")
    forward = np.asarray(
        validate_question_pairs(pairs_forward)
    )

    print("Scoring question pairs: reverse direction...")
    reverse = np.asarray(
        validate_question_pairs(pairs_reverse)
    )

    if (
        forward.ndim != 2
        or reverse.ndim != 2
        or forward.shape[1] < 3
        or reverse.shape[1] < 3
    ):
        raise ValueError(
            "Unexpected NLI output shape. Expected three probabilities: "
            "[contradiction, entailment, neutral]."
        )

    if len(forward) != len(dataset) or len(reverse) != len(dataset):
        raise ValueError("NLI output count does not match dataset size.")

    # Column 1 = entailment probability.
    forward_entailment = forward[:, 1]
    reverse_entailment = reverse[:, 1]

    # Strategy 1: average both directions.
    mean_scores = (
        forward_entailment + reverse_entailment
    ) / 2

    # Strategy 2: use the weaker direction.
    minimum_scores = np.minimum(
        forward_entailment,
        reverse_entailment
    )

    # Evaluate both strategies using frozen thresholds.
    mean_predictions, mean_metrics = evaluate_strategy(
        name="NLI MEAN",
        scores=mean_scores,
        threshold=MEAN_THRESHOLD,
        labels=labels,
    )

    minimum_predictions, minimum_metrics = evaluate_strategy(
        name="NLI MINIMUM",
        scores=minimum_scores,
        threshold=MINIMUM_THRESHOLD,
        labels=labels,
    )

    # --------------------------------------------------
    # Strategy comparison
    # --------------------------------------------------

    print(f"\n{'=' * 55}")
    print("STRATEGY COMPARISON")
    print(f"{'=' * 55}")

    print(
        f"{'Metric':<15}"
        f"{'NLI Mean':<15}"
        f"{'NLI Minimum':<15}"
    )

    print("-" * 45)

    for metric in ["precision", "recall", "f1"]:
        print(
            f"{metric.capitalize():<15}"
            f"{mean_metrics[metric]:<15.3f}"
            f"{minimum_metrics[metric]:<15.3f}"
        )

    # --------------------------------------------------
    # Individual pair results
    # --------------------------------------------------

    print(f"\n{'=' * 55}")
    print("INDIVIDUAL PAIR RESULTS")
    print(f"{'=' * 55}")

    for index, item in enumerate(dataset):
        expected = labels[index]

        mean_prediction = mean_predictions[index]
        minimum_prediction = minimum_predictions[index]

        mean_status = (
            "CORRECT" if mean_prediction == expected else "WRONG"
        )

        minimum_status = (
            "CORRECT" if minimum_prediction == expected else "WRONG"
        )

        print(f"\nID: {item['id']}")
        print(f"Category: {item.get('category', 'N/A')}")
        print(f"Question A: {item['question_a']}")
        print(f"Question B: {item['question_b']}")
        print(f"Expected: {'MATCH' if expected else 'NO MATCH'}")

        print(
            f"Forward entailment: "
            f"{forward_entailment[index]:.5f}"
        )

        print(
            f"Reverse entailment: "
            f"{reverse_entailment[index]:.5f}"
        )

        print(f"Mean score: {mean_scores[index]:.5f}")
        print(f"Minimum score: {minimum_scores[index]:.5f}")

        print(
            f"NLI Mean: "
            f"{'MATCH' if mean_prediction else 'NO MATCH'} "
            f"({mean_status})"
        )

        print(
            f"NLI Minimum: "
            f"{'MATCH' if minimum_prediction else 'NO MATCH'} "
            f"({minimum_status})"
        )


if __name__ == "__main__":
    main()
