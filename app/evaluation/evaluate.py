import json
from pathlib import Path

from app.retrieval.search import search_documents

BASE_DIR=Path(__file__).resolve().parents[2]
DATASET_PATH=BASE_DIR/"evaluation"/"eval_dataset.json"

TOP_K=3

def load_evaluation_dataset():
	with open(DATASET_PATH,"r",encoding="utf-8") as f:
		return json.load(f)

def evaluate_retrieval():
	dataset=load_evaluation_dataset()

	total_answerable=0
	successful_retrievals=0
	unanswerable_count=0

	print("\n==========RETRIEVAL EVALUATION========\n")

	for index,item in enumerate(dataset,start=1):
		question=item["question"]
		expected_chunk=item["expected_chunk"]
		answerable=item["answerable"]

		retrieved_docs=search_documents(question,limit=TOP_K)
		retrieved_chunks=[doc["chunk_index"] for doc in retrieved_docs]

		print(f"Test {index}:{question}")
		print("Retrieved chunks:",retrieved_chunks)

		if answerable:
			total_answerable+=1

			if expected_chunk in retrieved_chunks:
				successful_retrievals+=1
				print("Result: PASS")
			else:
				print("Result: FAIL")

		else:
			unanswerable_count+=1
			if retrieved_docs:
				print("Top similarity score:",
				      round(retrieved_docs[0]["score"],4))
			else:
				print("No documents retrieved.")
			print("Result: UNANSWERABLE QUESTION")

		print("-"*55)

	hit_rate=(
		successful_retrievals/total_answerable
	  if total_answerable else 0
		)
	print("\n=======EVALUATION SUMMARY=======\n")
	print("Total questions:",len(dataset))
	print("Answerable questions:",total_answerable)
	print("Unanaswerable questions:",unanswerable_count)
	print("Successful Retrievals:",successful_retrievals)
	print(f"Hit Rate @{TOP_K}: {hit_rate:.2%}")

if __name__ == "__main__":
	evaluate_retrieval()
