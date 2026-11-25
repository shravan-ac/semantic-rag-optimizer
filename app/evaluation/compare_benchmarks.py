
import csv
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
EVAL_DIR = BASE_DIR / "evaluation"

CONFIGS = {
    "Always-large baseline": {
        "results": "baseline_results.json",
        "quality": "baseline_answer_quality_results.csv",
    },
    "Routed generation (no cache)": {
        "results": "generation_only_results.json",
        "quality": "answer_quality_results.csv",
    },
    "Full optimization (warm cache)": {
        "results": "full_optimization_results.json",
        "quality": "full_optimization_quality_results.csv",
    },
}


def load_json(filename):
    path = EVAL_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_results(filename):
    data = load_json(filename)

    if isinstance(data, dict):
        data = data.get("results", [])

    if not isinstance(data, list):
        raise ValueError(f"{filename} must contain a list of results.")

    return data


def percentile(values, percentage):
    """Calculate a percentile using linear interpolation."""
    if not values:
        return None

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    position = (len(values) - 1) * percentage
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower

    return values[lower] + (values[upper] - values[lower]) * fraction


def calculate_quality(filename):
    path = EVAL_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    scores = []

    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError(f"{filename} has no CSV headers.")

        # Find the Token F1 column without assuming exact capitalization.
        f1_column = next(
            (
                column for column in reader.fieldnames
                if column.lower().replace(" ", "_") in {
                    "token_f1",
                    "f1",
                    "f1_score",
                    "token_f1_score",
                }
            ),
            None,
        )

        if f1_column is None:
            raise ValueError(
                f"Could not find a Token F1 column in {filename}. "
                f"Available columns: {reader.fieldnames}"
            )

        for row in reader:
            # Only score answerable questions.
            answerable_value = str(
                row.get("answerable", "")
            ).strip().lower()

            if answerable_value not in {"true", "1", "yes"}:
                continue

            try:
                score = float(row[f1_column])
            except (TypeError, ValueError):
                continue

            scores.append(score)

    return sum(scores) / len(scores) if scores else None


def calculate_metrics(results, quality):
    successful = [
        row for row in results
        if row.get("generation_status", "SUCCESS") == "SUCCESS"
        and not row.get("error")
    ]

    latencies = [
        float(row["latency_ms"])
        for row in successful
        if row.get("latency_ms") is not None
    ]

    costs = [
        float(row["estimated_cost_usd"])
        for row in successful
        if row.get("estimated_cost_usd") is not None
    ]

    cache_hits = sum(
        1 for row in successful
        if row.get("cache_hit") is True
    )

    total = len(results)

    return {
        "questions": total,
        "successful": len(successful),
        "quality": quality,
        "cost": sum(costs),
        "average_latency": (
            sum(latencies) / len(latencies) if latencies else None
        ),
        "p50_latency": percentile(latencies, 0.50),
        "p95_latency": percentile(latencies, 0.95),
        "cache_hits": cache_hits,
        "cache_hit_rate": (
            cache_hits / total * 100 if total else 0
        ),
    }


def format_number(value, digits=4):
    if value is None:
        return "N/A"

    return f"{value:.{digits}f}"


def main():
    metrics = {}

    for name, files in CONFIGS.items():
        results = load_results(files["results"])
        quality = calculate_quality(files["quality"])

        metrics[name] = calculate_metrics(results, quality)

    baseline_name = "Always-large baseline"
    baseline_cost = metrics[baseline_name]["cost"]
    baseline_quality = metrics[baseline_name]["quality"]

    print("\nBENCHMARK COMPARISON")
    print("=" * 90)

    headers = [
        "Metric",
        "Always-large",
        "Routed",
        "Full optimization",
    ]

    rows = [
        (
            "Answerable Token F1",
            "quality",
            4,
        ),
        (
            "Estimated cost (USD)",
            "cost",
            8,
        ),
        (
            "Average latency (ms)",
            "average_latency",
            2,
        ),
        (
            "p50 latency (ms)",
            "p50_latency",
            2,
        ),
        (
            "p95 latency (ms)",
            "p95_latency",
            2,
        ),
        (
            "Cache hit rate (%)",
            "cache_hit_rate",
            2,
        ),
    ]

    print(
        f"{headers[0]:<27}"
        f"{headers[1]:>18}"
        f"{headers[2]:>18}"
        f"{headers[3]:>24}"
    )
    print("-" * 90)

    names = list(CONFIGS.keys())

    for label, key, digits in rows:
        values = [
            format_number(metrics[name][key], digits)
            for name in names
        ]

        print(
            f"{label:<27}"
            f"{values[0]:>18}"
            f"{values[1]:>18}"
            f"{values[2]:>24}"
        )

    print("\nCOST REDUCTION")
    print("-" * 40)

    for name in names[1:]:
        cost = metrics[name]["cost"]

        if baseline_cost:
            reduction = (baseline_cost - cost) / baseline_cost * 100
            print(f"{name}: {reduction:.2f}% vs always-large baseline")

    print("\nQUALITY CHANGE")
    print("-" * 40)

    for name in names[1:]:
        quality = metrics[name]["quality"]

        if baseline_quality is not None and quality is not None:
            difference = quality - baseline_quality
            print(
                f"{name}: {difference:+.4f} Token F1 "
                "vs always-large baseline"
            )

    print("\nRUN STATUS")
    print("-" * 40)

    for name in names:
        item = metrics[name]
        print(
            f"{name}: {item['successful']}/{item['questions']} successful"
        )

    print("\nIMPORTANT NOTES")
    print("-" * 40)
    print("- Cost values are estimates, not actual provider bills.")
    print("- Full optimization uses a pre-warmed cache.")
    print("- Token F1 measures lexical overlap, not factual correctness.")
    print("- Results use the existing 10-question evaluation dataset.")

    print("\nNo LLM calls were made. Existing benchmark files were only read.")


if __name__ == "__main__":
    main()
