
import json
from pathlib import Path

from app.rag import ask_question
from app.retrieval.search import search_documents


BASE_DIR = Path(__file__).resolve().parents[2]

DATASET_PATH = BASE_DIR / "evaluation" / "eval_dataset.json"
RESULTS_PATH = BASE_DIR / "evaluation" / "answer_results.json"

TOP_K = 3


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def detect_refusal(answer):
    """
    Detect common phrases indicating that the documents
    do not contain the requested information.
    """
    answer_lower = answer.lower()

    refusal_phrases = [
        "couldn't find this information",
        "couldn't find relevant information",
        "do not specify",
        "does not specify",
        "do not contain",
        "does not contain",
        "not mentioned in the documents",
        "not available in the provided documents",
        "no relevant information",
    ]

    return any(
        phrase in answer_lower
        for phrase in refusal_phrases
    )


def evaluate_answers():
    dataset = load_dataset()
    results = []

    print("\n========== ANSWER EVALUATION ==========\n")

    for index, item in enumerate(dataset, start=1):
        question = item["question"]
        expected_answer = item["expected_answer"]
        expected_chunk = item["expected_chunk"]
        answerable = item["answerable"]

        print(f"\nTest {index}/{len(dataset)}")
        print("Question:", question)

        try:
            # Evaluate retrieval.
            retrieved_docs = search_documents(
                question,
                limit=TOP_K
            )

            retrieved_chunks = [
                doc["chunk_index"] for doc in retrieved_docs
            ]

            retrieval_hit = (
                expected_chunk in retrieved_chunks
                if answerable else None
            )

            # Generate an answer through the RAG pipeline.
            response = ask_question(question)

            answer = response["answer"]
            refusal = detect_refusal(answer)

            if answerable:
                review_status = "MANUAL_REVIEW"
            else:
                review_status = (
                    "REFUSAL_PASS"
                    if refusal
                    else "MANUAL_REVIEW"
                )

            result = {
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": answer,
                "answerable": answerable,
                "expected_chunk": expected_chunk,
                "retrieved_chunks": retrieved_chunks,
                "retrieval_hit": retrieval_hit,
                "refusal_detected": refusal,
                "review_status": review_status,
                "generation_status": "SUCCESS",
                "model": response["model"],
                "prompt_tokens": response["prompt_tokens"],
                "completion_tokens": response["completion_tokens"],
                "total_tokens": response["total_tokens"],
                "latency_ms": response["latency_ms"]
            }

            results.append(result)

            print("Retrieved chunks:", retrieved_chunks)
            print("Generated answer:", answer)
            print("Expected answer:", expected_answer)
            print("Retrieval hit:", retrieval_hit)
            print("Refusal detected:", refusal)
            print("Review status:", review_status)
            print("Tokens:", response["total_tokens"])
            print("Latency:", response["latency_ms"], "ms")

        except Exception as error:
            print("ERROR:", str(error))

            results.append({
                "question": question,
                "expected_answer": expected_answer,
                "answerable": answerable,
                "expected_chunk": expected_chunk,
                "generation_status": "ERROR",
                "error": str(error),
                "review_status": "ERROR"
            })

    # Save all results, including generation errors.
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False
        )

    # Calculate evaluation counts.
    successful = [
        result for result in results
        if result["generation_status"] == "SUCCESS"
    ]

    errors = [
        result for result in results
        if result["generation_status"] == "ERROR"
    ]

    answerable_total = sum(
        item["answerable"] for item in dataset
    )

    unanswerable_total = len(dataset) - answerable_total

    answerable_success = [
        result for result in successful
        if result["answerable"]
    ]

    unanswerable_success = [
        result for result in successful
        if not result["answerable"]
    ]

    answerable_errors = [
        result for result in errors
        if result["answerable"]
    ]

    unanswerable_errors = [
        result for result in errors
        if not result["answerable"]
    ]

    refusal_passes = sum(
        result["review_status"] == "REFUSAL_PASS"
        for result in unanswerable_success
    )

    print("\n========== EVALUATION SUMMARY ==========")

    print("Total questions:", len(dataset))
    print("Successful generations:", len(successful))
    print("Generation errors:", len(errors))

    print("\nANSWERABLE QUESTIONS")
    print("Total:", answerable_total)
    print("Successful:", len(answerable_success))
    print("Errors:", len(answerable_errors))

    print("\nUNANSWERABLE QUESTIONS")
    print("Total:", unanswerable_total)
    print("Successful:", len(unanswerable_success))
    print("Errors:", len(unanswerable_errors))
    print("Correct refusal detections:", refusal_passes)

    print("\nResults saved to:", RESULTS_PATH)


if __name__ == "__main__":
    evaluate_answers()
