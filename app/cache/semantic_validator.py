import numpy as np
import re
from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

NLI_MEAN_THRESHOLD = 0.27458

_validator = None


def get_validator():
    global _validator

    if _validator is None:
        print("Loading local NLI model...")

        _validator = CrossEncoder(
            MODEL_NAME,
            max_length=256
        )

        print("NLI model loaded.")

    return _validator


def validate_question_pairs(pairs: list[tuple[str, str]]):
    if not pairs:
        return []

    model = get_validator()

    predictions = model.predict(
        pairs,
        apply_softmax=True,
        show_progress_bar=False
    )

    return predictions


def score_question_pairs_bidirectional(
    pairs: list[tuple[str, str]]
):
    """
    Calculate mean entailment probability in both directions.

    NLI output columns:
    0 = contradiction
    1 = entailment
    2 = neutral
    """

    if not pairs:
        return []

    forward_predictions = validate_question_pairs(pairs)

    reverse_pairs = [
        (question_b, question_a)
        for question_a, question_b in pairs
    ]

    reverse_predictions = validate_question_pairs(reverse_pairs)

    forward_predictions = np.asarray(forward_predictions)
    reverse_predictions = np.asarray(reverse_predictions)

    if (
        forward_predictions.ndim != 2
        or reverse_predictions.ndim != 2
        or forward_predictions.shape[1] < 3
        or reverse_predictions.shape[1] < 3
    ):
        raise ValueError("Unexpected NLI prediction output.")

    forward_entailment = forward_predictions[:, 1]
    reverse_entailment = reverse_predictions[:, 1]

    mean_scores = (
        forward_entailment + reverse_entailment
    ) / 2

    return mean_scores.tolist()

def has_explicit_negation(text: str) -> bool:
    """
    Detect common explicit negation expressions.
    This is a lightweight safeguard, not a complete
    natural-language contradiction detector.
    """
    patterns = [
        r"\bnot\b",
        r"\bno\b",
        r"\bnever\b",
        r"\bcannot\b",
        r"\bcan't\b",
        r"\bdon't\b",
        r"\bdoesn't\b",
        r"\bdidn't\b",
        r"\bwon't\b",
        r"\bwouldn't\b",
        r"\bshouldn't\b",
        r"\bmustn't\b",
        r"\bisn't\b",
        r"\baren't\b",
        r"\bwasn't\b",
        r"\bweren't\b",
        r"\bneither\b",
        r"\bprohibited\b",
        r"\bforbidden\b",
    ]

    normalized_text = text.lower()

    return any(
        re.search(pattern, normalized_text)
        for pattern in patterns
    )

def are_questions_equivalent(question_a: str, question_b: str):

    scores = score_question_pairs_bidirectional(
        [(question_a, question_b)]
    )

    score = scores[0]

    return {
        "equivalent": score >= NLI_MEAN_THRESHOLD,
        "score": float(score),
        "threshold": NLI_MEAN_THRESHOLD
    }
