
import json
from pathlib import Path

import numpy as np

from app.ingestion.model import embedding_model
from app.cache.semantic_validator import validate_question_pairs


BASE_DIR = Path(__file__).resolve().parents[2]

DATASET_PATH = BASE_DIR / "app" / "evaluation" / "cache_pairs.json"
RESULT_PATH = BASE_DIR / "app" / "evaluation" / "cache_model_comparison.json"


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    for item in dataset:
        if "question_a" not in item or "question_b" not in item:
            raise ValueError(
                f"Pair {item.get('id', 'unknown')} is missing question_a or question_b."
            )

        if "equivalent" not in item:
            raise ValueError(
                f"Pair {item.get('id', 'unknown')} is missing equivalent."
            )

    return dataset


def calculate_metrics(scores, labels, threshold):
    predictions = [score >= threshold for score in scores]

    tp = sum(p and y for p, y in zip(predictions, labels))
    fp = sum(p and not y for p, y in zip(predictions, labels))
    tn = sum(not p and not y for p, y in zip(predictions, labels))
    fn = sum(not p and y for p, y in zip(predictions, labels))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def find_best_threshold(scores, labels):
    unique_scores = sorted(set(float(s) for s in scores))

    if not unique_scores:
        raise ValueError("No scores were generated.")

    candidates = [unique_scores[0] - 0.000001]

    for left, right in zip(unique_scores, unique_scores[1:]):
        candidates.append((left + right) / 2)

    candidates.append(unique_scores[-1] + 0.000001)

    results = [
        calculate_metrics(scores, labels, threshold)
        for threshold in candidates
    ]

    # Prefer higher F1, then higher precision, then higher threshold.
    return max(
        results,
        key=lambda result: (
            result["f1"],
            result["precision"],
            result["threshold"],
        ),
    )


def evaluate_bge(dataset):
    print("\nEvaluating BGE embedding similarity...")

    questions = []
    for item in dataset:
        questions.extend([item["question_a"], item["question_b"]])

    embeddings = embedding_model.encode(
        questions,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    scores = []

    for index in range(0, len(embeddings), 2):
        # Normalized embeddings: dot product is cosine similarity.
        score = float(np.dot(embeddings[index], embeddings[index + 1]))
        scores.append(score)

    return scores


def evaluate_minilm(dataset):
    print("\nEvaluating MiniLM cross-encoder...")

    forward_pairs = [
        (item["question_a"], item["question_b"])
        for item in dataset
    ]

    reverse_pairs = [
        (item["question_b"], item["question_a"])
        for item in dataset
    ]

    forward_scores = score_question_pairs(forward_pairs)
    reverse_scores = score_question_pairs(reverse_pairs)

    # Use the average to reduce sensitivity to question order.
    scores = [
        (float(forward) + float(reverse)) / 2
        for forward, reverse in zip(forward_scores, reverse_scores)
    ]

    return scores


def evaluate_nli(dataset):
    print("\nEvaluating DeBERTa NLI...")

    forward_pairs = [
        (item["question_a"], item["question_b"])
        for item in dataset
    ]

    reverse_pairs = [
        (item["question_b"], item["question_a"])
        for item in dataset
    ]

    forward_predictions = np.asarray(
        validate_question_pairs(forward_pairs)
    )

    reverse_predictions = np.asarray(
        validate_question_pairs(reverse_pairs)
    )

    if (
        forward_predictions.ndim != 2
        or reverse_predictions.ndim != 2
        or forward_predictions.shape[1] != 3
        or reverse_predictions.shape[1] != 3
    ):
        raise ValueError(
            "Expected NLI predictions with three columns: "
            "[contradiction, entailment, neutral]. "
            "Check the model output and class order."
        )

    # Column 1 is entailment, based on the current validator implementation.
    forward_entailment = forward_predictions[:, 1]
    reverse_entailment = reverse_predictions[:, 1]

    # Keep both methods for comparison:
    # min is conservative; mean is less strict about direction.
    minimum_scores = np.minimum(
        forward_entailment,
        reverse_entailment,
    )

    mean_scores = (
        forward_entailment + reverse_entailment
    ) / 2

    return {
        "NLI bidirectional minimum": minimum_scores.tolist(),
        "NLI bidirectional mean": mean_scores.tolist(),
    }


def main():
    dataset = load_dataset()

    labels = [bool(item["equivalent"]) for item in dataset]

    positive_count = sum(labels)
    negative_count = len(labels) - positive_count

    print(f"Loaded {len(dataset)} question pairs.")
    print(f"Equivalent pairs: {positive_count}")
    print(f"Non-equivalent pairs: {negative_count}")

    if positive_count == 0 or negative_count == 0:
        raise ValueError(
            "The dataset must contain both equivalent and non-equivalent pairs."
        )

    all_scores = {
        "BGE cosine similarity": evaluate_bge(dataset),
        "MiniLM bidirectional mean": evaluate_minilm(dataset),
    }

    all_scores.update(evaluate_nli(dataset))

    detailed_results = []

    for index, item in enumerate(dataset):
        row = {
            "id": item["id"],
            "category": item.get("category"),
            "question_a": item["question_a"],
            "question_b": item["question_b"],
            "equivalent": labels[index],
            "scores": {
                model_name: float(scores[index])
                for model_name, scores in all_scores.items()
            },
        }

        detailed_results.append(row)

    model_summaries = {}

    print("\n========== MODEL COMPARISON ==========")

    for model_name, scores in all_scores.items():
        best = find_best_threshold(scores, labels)

        model_summaries[model_name] = {
            "best_metrics": best,
        }

        print(f"\n{model_name}")
        print(f"Best threshold: {best['threshold']:.5f}")
        print(f"Precision:      {best['precision']:.3f}")
        print(f"Recall:         {best['recall']:.3f}")
        print(f"F1:             {best['f1']:.3f}")
        print(f"TP: {best['tp']} | FP: {best['fp']}")
        print(f"TN: {best['tn']} | FN: {best['fn']}")

        # Record which pairs are misclassified at this model's best threshold.
        for index, score in enumerate(scores):
            predicted = float(score) >= best["threshold"]

            if predicted != labels[index]:
                detailed_results[index].setdefault(
                    "misclassified_by", []
                ).append(model_name)

    output = {
        "dataset_path": str(DATASET_PATH),
        "pair_count": len(dataset),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "model_summaries": model_summaries,
        "pairs": detailed_results,
    }

    with open(RESULT_PATH, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4, ensure_ascii=False)

    print("\nDetailed results saved to:")
    print(RESULT_PATH)


if __name__ == "__main__":
    main()
