
import json
from pathlib import Path

import pandas as pd
import streamlit as st


# --------------------------------------------------
# Paths and benchmark configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
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


# --------------------------------------------------
# Data loading
# --------------------------------------------------

@st.cache_data
def load_json(filename):
    path = EVAL_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        data = data.get("results", [])

    if not isinstance(data, list):
        raise ValueError(f"{filename} must contain a list.")

    return data


@st.cache_data
def load_csv(filename):
    path = EVAL_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    return pd.read_csv(path)


def get_quality_score(filename):
    df = load_csv(filename)

    normalized_columns = {
        str(column).lower().replace(" ", "_"): column
        for column in df.columns
    }

    f1_column = next(
        (
            normalized_columns[name]
            for name in (
                "token_f1",
                "f1",
                "f1_score",
                "token_f1_score",
            )
            if name in normalized_columns
        ),
        None,
    )

    if f1_column is None:
        raise ValueError(
            f"Token F1 column not found in {filename}. "
            f"Columns: {list(df.columns)}"
        )

    answerable_column = normalized_columns.get("answerable")

    if answerable_column:
        answerable = (
            df[answerable_column]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
        )
        df = df[answerable]

    scores = pd.to_numeric(df[f1_column], errors="coerce").dropna()

    return float(scores.mean()) if not scores.empty else None


def calculate_percentile(values, percentile):
    if not values:
        return None

    return float(pd.Series(values).quantile(percentile))


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
        "p50_latency": calculate_percentile(latencies, 0.50),
        "p95_latency": calculate_percentile(latencies, 0.95),
        "cache_hits": cache_hits,
        "cache_hit_rate": (
            cache_hits / total * 100 if total else 0
        ),
    }


# --------------------------------------------------
# Load all benchmark data
# --------------------------------------------------

st.set_page_config(
    page_title="Semantic RAG Optimizer",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Semantic Caching & Cost-Aware Routing")
st.caption("Benchmark dashboard — RAG performance, answer quality and cost")

try:
    all_results = {}
    metrics = {}

    for name, files in CONFIGS.items():
        results = load_json(files["results"])
        quality = get_quality_score(files["quality"])

        all_results[name] = results
        metrics[name] = calculate_metrics(results, quality)

except (FileNotFoundError, ValueError, KeyError) as error:
    st.error(f"Could not load benchmark data: {error}")
    st.stop()


# --------------------------------------------------
# Overview
# --------------------------------------------------

baseline = metrics["Always-large baseline"]
routed = metrics["Routed generation (no cache)"]
optimized = metrics["Full optimization (warm cache)"]

st.header("Benchmark overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Optimized answer quality",
        f"{optimized['quality']:.4f}"
        if optimized["quality"] is not None else "N/A",
    )

with col2:
    if baseline["cost"] > 0:
        reduction = (
            baseline["cost"] - optimized["cost"]
        ) / baseline["cost"] * 100
        st.metric("Cost reduction", f"{reduction:.2f}%")
    else:
        st.metric("Cost reduction", "N/A")

with col3:
    st.metric(
        "Cache hit rate",
        f"{optimized['cache_hit_rate']:.1f}%",
    )

with col4:
    st.metric(
        "Successful runs",
        f"{optimized['successful']}/{optimized['questions']}",
    )


# --------------------------------------------------
# Configuration comparison
# --------------------------------------------------

st.header("Configuration comparison")

comparison = pd.DataFrame(
    {
        name: {
            "Answerable Token F1": item["quality"],
            "Estimated cost (USD)": item["cost"],
            "Average latency (ms)": item["average_latency"],
            "p50 latency (ms)": item["p50_latency"],
            "p95 latency (ms)": item["p95_latency"],
            "Cache hit rate (%)": item["cache_hit_rate"],
            "Successful generations": (
                f"{item['successful']}/{item['questions']}"
            ),
        }
        for name, item in metrics.items()
    }
)

st.dataframe(
    comparison,
    use_container_width=True,
)


# --------------------------------------------------
# Charts
# --------------------------------------------------

st.header("Performance visualizations")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Answer quality")
    quality_data = pd.DataFrame(
        {
            "Configuration": list(metrics.keys()),
            "Token F1": [
                item["quality"] for item in metrics.values()
            ],
        }
    ).set_index("Configuration")

    st.bar_chart(quality_data)

with chart_col2:
    st.subheader("Estimated generation cost")
    cost_data = pd.DataFrame(
        {
            "Configuration": list(metrics.keys()),
            "Cost (USD)": [
                item["cost"] for item in metrics.values()
            ],
        }
    ).set_index("Configuration")

    st.bar_chart(cost_data)


latency_data = pd.DataFrame(
    {
        "Average latency (ms)": {
            name: item["average_latency"]
            for name, item in metrics.items()
        },
        "p50 latency (ms)": {
            name: item["p50_latency"]
            for name, item in metrics.items()
        },
        "p95 latency (ms)": {
            name: item["p95_latency"]
            for name, item in metrics.items()
        },
    }
)

st.subheader("Latency comparison")
st.bar_chart(latency_data)


# --------------------------------------------------
# Cost savings
# --------------------------------------------------

st.header("Cost savings")

if baseline["cost"] > 0:
    routed_saving = (
        baseline["cost"] - routed["cost"]
    ) / baseline["cost"] * 100

    optimized_saving = (
        baseline["cost"] - optimized["cost"]
    ) / baseline["cost"] * 100

    col1, col2 = st.columns(2)

    col1.metric(
        "Routing vs baseline",
        f"{routed_saving:.2f}%",
    )

    col2.metric(
        "Full optimization vs baseline",
        f"{optimized_saving:.2f}%",
    )


# --------------------------------------------------
# Question-level results
# --------------------------------------------------

st.header("Question-level benchmark results")

selected_config = st.selectbox(
    "Select configuration",
    list(CONFIGS.keys()),
)

selected_results = all_results[selected_config]

question_rows = []

for row in selected_results:
    question_rows.append(
        {
            "Question": row.get("question", ""),
            "Answer": row.get("answer", ""),
            "Model": row.get("model", ""),
            "Route": row.get("route", ""),
            "Cache hit": row.get("cache_hit", False),
            "Latency (ms)": row.get("latency_ms"),
            "Estimated cost (USD)": row.get("estimated_cost_usd"),
            "Status": row.get("generation_status", "UNKNOWN"),
        }
    )

question_df = pd.DataFrame(question_rows)

st.dataframe(
    question_df,
    use_container_width=True,
    hide_index=True,
)


# --------------------------------------------------
# Individual answer inspection
# --------------------------------------------------

st.subheader("Inspect an individual answer")

if selected_results:
    question_options = [
        row.get("question", f"Question {index + 1}")
        for index, row in enumerate(selected_results)
    ]

    selected_question = st.selectbox(
        "Choose a question",
        question_options,
        key="answer_inspection",
    )

    selected_row = next(
        row for row in selected_results
        if row.get("question") == selected_question
    )

    st.markdown("**Question**")
    st.write(selected_row.get("question", ""))

    st.markdown("**Generated answer**")
    st.write(selected_row.get("answer", "No answer available."))

    st.markdown("**Expected answer**")
    st.write(selected_row.get("expected_answer", "Not available."))

    st.markdown("**Execution details**")
    st.json(
        {
            "model": selected_row.get("model"),
            "route": selected_row.get("route"),
            "cache_hit": selected_row.get("cache_hit"),
            "cache_similarity": selected_row.get("cache_similarity"),
            "cache_nli_score": selected_row.get("cache_nli_score"),
            "latency_ms": selected_row.get("latency_ms"),
            "estimated_cost_usd": selected_row.get(
                "estimated_cost_usd"
            ),
            "generation_status": selected_row.get(
                "generation_status"
            ),
        }
    )


# --------------------------------------------------
# Important notes
# --------------------------------------------------

st.divider()

st.subheader("Benchmark notes")

st.info(
    """
    - Full optimization uses a pre-warmed cache with curated entries.
    - Estimated costs are estimates, not actual provider bills.
    - Token F1 measures lexical overlap, not factual correctness.
    - Results use the existing 10-question evaluation dataset.
    - This dashboard reads existing benchmark files and makes no LLM calls.
    """
)

st.caption("Semantic Caching & Cost-Aware Routing for RAG")
