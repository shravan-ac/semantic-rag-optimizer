
import json
from pathlib import Path
from time import perf_counter

from app.retrieval.search import search_documents
from app.generation.generator import generate_answer
from app.routing.router import route_query
from app.analytics.pricing import estimate_cost


# Resolve project root
BASE_DIR = Path(__file__).resolve().parents[2]

DATASET_PATH = BASE_DIR / "evaluation" / "eval_dataset.json"
OUTPUT_PATH = BASE_DIR / "evaluation" / "generation_only_results.json"

# Start with 3 questions to limit API usage.
# Change to None when ready to evaluate the full dataset.
MAX_QUESTIONS = None


def evaluate_generation():

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    if MAX_QUESTIONS is not None:
        dataset = dataset[:MAX_QUESTIONS]

    results = []

    successful = 0
    errors = 0
    total_tokens = 0
    total_cost = 0.0
    total_latency = 0.0
    known_cost_count = 0

    print("\nCACHE-INDEPENDENT GENERATION EVALUATION")
    print("=" * 50)
    print("Questions:", len(dataset))
    print("Semantic cache: DISABLED")

    for index, item in enumerate(dataset, start=1):

        question = item["question"]
        expected_answer = item.get("expected_answer")
        answerable = item.get("answerable")
        expected_chunk = item.get("expected_chunk")

        print(f"\nQuestion {index}/{len(dataset)}")
        print("Question:", question)

        start_time = perf_counter()

        try:
            # 1. Retrieve relevant document chunks
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

            # 2. Select model using the cost-aware router
            routing_decision = route_query(question)

            selected_model = routing_decision["model"]

            print("Selected model:", selected_model)
            print("Selected route:", routing_decision["route"])

            # 3. Generate a fresh answer.
            # This directly calls the generator.
            # No semantic cache lookup or cache save occurs.
            generation = generate_answer(
                question,
                retrieved_docs,
                model_name=selected_model
            )

            latency_ms = (perf_counter() - start_time) * 1000

            estimated_cost = estimate_cost(
                generation["prompt_tokens"],
                generation["completion_tokens"],
                model=generation["model"]
            )

            answer = generation["answer"]

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
                "generation_status": "SUCCESS"
            }

            successful += 1
            total_tokens += generation["total_tokens"]
            total_latency += latency_ms

            if estimated_cost is not None:
                total_cost += estimated_cost
                known_cost_count += 1

            print("Answer:", answer)
            print("Tokens:", generation["total_tokens"])
            print("Latency (ms):", round(latency_ms, 2))
            print("Estimated cost:", estimated_cost)

        except Exception as error:

            latency_ms = (perf_counter() - start_time) * 1000

            result = {
                "question": question,
                "expected_answer": expected_answer,
                "answerable": answerable,
                "expected_chunk": expected_chunk,
                "answer": None,
                "error": str(error),
                "latency_ms": round(latency_ms, 2),
                "cache_hit": False,
                "generation_status": "ERROR"
            }

            errors += 1
            print("Generation error:", error)

        results.append(result)

    # Save results
    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("GENERATION EVALUATION SUMMARY")
    print("=" * 50)

    print("Total questions:", len(dataset))
    print("Successful generations:", successful)
    print("Generation errors:", errors)
    print("Total tokens:", total_tokens)

    if successful:
        print(
            "Average latency (ms):",
            round(total_latency / successful, 2)
        )

    print("Questions with known cost:", known_cost_count)
    print("Total estimated cost (USD):", round(total_cost, 8))

    print("\nResults saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    evaluate_generation()
