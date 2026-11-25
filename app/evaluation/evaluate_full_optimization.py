
import json
import uuid
from pathlib import Path
from time import perf_counter

from app.cache.semantic_cache import (
    create_cache_collection,
    search_cache,
    save_to_cache,
    client as qdrant_client,
)
from app.retrieval.search import search_documents
from app.routing.router import route_query
from app.generation.generator import generate_answer
from app.analytics.pricing import estimate_cost


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

DATASET_PATH = BASE_DIR / "evaluation" / "eval_dataset.json"
OUTPUT_PATH = BASE_DIR / "evaluation" / "full_optimization_results.json"

# Run a 3-question smoke test first.
# Change to None to run the full evaluation dataset.
MAX_QUESTIONS = None

# Each run gets its own isolated collection.
EXPERIMENT_COLLECTION = (
    f"semantic_cache_benchmark_{uuid.uuid4().hex[:8]}"
)


# --------------------------------------------------
# Cache warm-up data
# These answers are grounded in leave_policy.txt.
# They are inserted without making LLM calls.
# --------------------------------------------------

WARMUP_DATA = [
    {
        "question": "How many paid annual leave days are provided each year?",
        "answer": "Employees receive 20 paid annual leave days per year.",
    },
    {
        "question": "How early must an employee apply for annual leave?",
        "answer": (
            "Employees should submit leave requests at least "
            "five working days in advance."
        ),
    },
    {
        "question": "Who approves leave applications?",
        "answer": "The reporting manager approves leave requests.",
    },
    {
        "question": "What is the annual leave carry-forward limit?",
        "answer": (
            "Employees may carry forward up to five unused annual "
            "leave days to the next calendar year."
        ),
    },
    {
        "question": "How many paid sick leave days are provided each year?",
        "answer": "Employees receive 10 paid sick leave days per year.",
    },
    {
        "question": "When can medical documentation be requested?",
        "answer": (
            "Medical documentation may be requested for absences "
            "longer than three consecutive working days."
        ),
    },
    {
        "question": "When can emergency leave be requested?",
        "answer": (
            "Employees may request emergency leave in unexpected "
            "personal situations; approval depends on the circumstances."
        ),
    },
]


def warm_up_cache():
    """Insert known policy answers into the isolated experiment cache."""

    print("\nWARMING UP EXPERIMENT CACHE")
    print("=" * 50)

    for item in WARMUP_DATA:
        save_to_cache(
            query=item["question"],
            answer=item["answer"],
            sources=["leave_policy.txt"],
            collection_name=EXPERIMENT_COLLECTION,
        )

    print("Warm-up entries:", len(WARMUP_DATA))


def percentile(values, percent):
    """Calculate a simple nearest-rank percentile."""

    if not values:
        return None

    ordered = sorted(values)
    index = max(0, int((percent / 100) * len(ordered) + 0.999999) - 1)

    return round(ordered[index], 2)


def evaluate_full_optimization():

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    if MAX_QUESTIONS is not None:
        dataset = dataset[:MAX_QUESTIONS]

    results = []

    cache_hits = 0
    cache_misses = 0
    successful = 0
    errors = 0

    total_tokens = 0
    total_cost = 0.0
    known_cost_count = 0

    latencies = []

    collection_created = False

    print("\nFULL OPTIMIZATION EVALUATION")
    print("=" * 50)
    print("Questions:", len(dataset))
    print("Experiment collection:", EXPERIMENT_COLLECTION)
    print("Production cache: NOT USED")
    print("SQLite analytics: NOT USED")

    try:
        # 1. Create a new isolated cache collection.
        create_cache_collection(
            collection_name=EXPERIMENT_COLLECTION
        )
        collection_created = True

        # 2. Insert known answers into the experiment cache.
        warm_up_cache()

        # 3. Evaluate the dataset.
        for index, item in enumerate(dataset, start=1):

            question = item["question"]
            expected_answer = item.get("expected_answer")
            answerable = item.get("answerable")
            expected_chunk = item.get("expected_chunk")

            print(f"\nQuestion {index}/{len(dataset)}")
            print("Question:", question)

            start_time = perf_counter()

            try:
                # First: semantic cache lookup.
                cached_result = search_cache(
                    question,
                    collection_name=EXPERIMENT_COLLECTION,
                )

                if cached_result:

                    # CACHE HIT: no retrieval or LLM call.
                    cache_hits += 1
                    successful += 1

                    answer = cached_result["answer"]
                    sources = cached_result["sources"]

                    latency_ms = (
                        perf_counter() - start_time
                    ) * 1000

                    result = {
                        "question": question,
                        "expected_answer": expected_answer,
                        "answerable": answerable,
                        "expected_chunk": expected_chunk,
                        "retrieved_chunks": [],
                        "retrieval_hit": None,
                        "answer": answer,
                        "sources": sources,
                        "model": "semantic-cache",
                        "route": "semantic-cache",
                        "complexity": "cached",
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "latency_ms": round(latency_ms, 2),
                        "estimated_cost_usd": 0.0,
                        "cache_hit": True,
                        "cache_similarity": cached_result.get("score"),
                        "cache_nli_score": cached_result.get("nli_score"),
                        "generation_status": "SUCCESS",
                    }

                    print("Cache hit: YES")
                    print("NLI score:", cached_result.get("nli_score"))
                    print("Answer:", answer)

                else:

                    # CACHE MISS: continue through normal RAG flow.
                    cache_misses += 1

                    print("Cache hit: NO")

                    # Retrieve relevant document chunks.
                    retrieved_docs = search_documents(question)

                    sources = list(dict.fromkeys(
                        doc["document_name"]
                        for doc in retrieved_docs
                    ))

                    retrieved_chunks = [
                        doc.get("chunk_index")
                        for doc in retrieved_docs
                    ]

                    retrieval_hit = (
                        expected_chunk in retrieved_chunks
                        if expected_chunk is not None
                        else None
                    )

                    # Select model using the cost-aware router.
                    routing_decision = route_query(question)
                    selected_model = routing_decision["model"]

                    print("Selected model:", selected_model)
                    print("Selected route:", routing_decision["route"])

                    # Generate answer.
                    generation = generate_answer(
                        question,
                        retrieved_docs,
                        model_name=selected_model,
                    )

                    answer = generation["answer"]

                    estimated_cost = estimate_cost(
                        generation["prompt_tokens"],
                        generation["completion_tokens"],
                        model=generation["model"],
                    )

                    # Save only answers for answerable questions.
                    # Do not cache unsupported refusals.
                    if answerable and answer and sources:
                        save_to_cache(
                            query=question,
                            answer=answer,
                            sources=sources,
                            collection_name=EXPERIMENT_COLLECTION,
                        )

                    latency_ms = (
                        perf_counter() - start_time
                    ) * 1000

                    result = {
                        "question": question,
                        "expected_answer": expected_answer,
                        "answerable": answerable,
                        "expected_chunk": expected_chunk,
                        "retrieved_chunks": retrieved_chunks,
                        "retrieval_hit": retrieval_hit,
                        "answer": answer,
                        "sources": sources,
                        "model": generation["model"],
                        "route": routing_decision["route"],
                        "complexity": routing_decision["complexity"],
                        "prompt_tokens": generation["prompt_tokens"],
                        "completion_tokens": generation["completion_tokens"],
                        "total_tokens": generation["total_tokens"],
                        "latency_ms": round(latency_ms, 2),
                        "estimated_cost_usd": estimated_cost,
                        "cache_hit": False,
                        "cache_similarity": None,
                        "cache_nli_score": None,
                        "generation_status": "SUCCESS",
                    }

                    successful += 1
                    total_tokens += generation["total_tokens"]

                    if estimated_cost is not None:
                        total_cost += estimated_cost
                        known_cost_count += 1

                    print("Answer:", answer)
                    print("Retrieval hit:", retrieval_hit)
                    print("Tokens:", generation["total_tokens"])
                    print("Estimated cost:", estimated_cost)

                latencies.append(latency_ms)
                results.append(result)

                print("Latency (ms):", round(latency_ms, 2))

            except Exception as error:

                latency_ms = (
                    perf_counter() - start_time
                ) * 1000

                latencies.append(latency_ms)

                results.append({
                    "question": question,
                    "expected_answer": expected_answer,
                    "answerable": answerable,
                    "expected_chunk": expected_chunk,
                    "answer": None,
                    "error": str(error),
                    "latency_ms": round(latency_ms, 2),
                    "cache_hit": False,
                    "generation_status": "ERROR",
                })

                errors += 1
                print("Evaluation error:", error)

        # Save the result list in the same format as other evaluators.
        with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=4, ensure_ascii=False)

        # Summary
        print("\n" + "=" * 50)
        print("FULL OPTIMIZATION SUMMARY")
        print("=" * 50)

        print("Total questions:", len(dataset))
        print("Successful:", successful)
        print("Errors:", errors)
        print("Cache hits:", cache_hits)
        print("Cache misses:", cache_misses)

        if dataset:
            print(
                "Cache hit rate (%):",
                round(cache_hits / len(dataset) * 100, 2),
            )

        print("Total generated tokens:", total_tokens)
        print("Total estimated generation cost (USD):", round(total_cost, 8))
        print("Questions with known cost:", known_cost_count)

        if latencies:
            print("Average latency (ms):", round(sum(latencies) / len(latencies), 2))
            print("p50 latency (ms):", percentile(latencies, 50))
            print("p95 latency (ms):", percentile(latencies, 95))

        print("\nResults saved to:", OUTPUT_PATH)

    finally:
        # Delete only the temporary collection created by this run.
        # Production collections are never touched.
        if collection_created:
            qdrant_client.delete_collection(
                collection_name=EXPERIMENT_COLLECTION
            )
            print(
                "\nTemporary experiment collection deleted:",
                EXPERIMENT_COLLECTION,
            )


if __name__ == "__main__":
    evaluate_full_optimization()
