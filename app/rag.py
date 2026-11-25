#RAG pipeline

from time import perf_counter

from app.retrieval.search import search_documents
from app.generation.generator import generate_answer
from app.analytics.logger import log_request
from app.analytics.pricing import estimate_cost

from app.cache.semantic_cache import search_cache,save_to_cache

from app.routing.router import route_query

def ask_question(question:str):
	if not question.strip():
		raise ValueError("Question cannot be empty")
	start_time=perf_counter()

	#a.check semantic cache firstly
	cached_result=search_cache(question)

	if cached_result:
		latency_ms=(perf_counter()-start_time)*1000
		log_request(
			question=question,
			model="semantic-cache",
			prompt_tokens=0,
			completion_tokens=0,
			total_tokens=0,
			latency_ms=latency_ms,
			estimated_cost_usd=0.0,
			cache_hit=True,
			route="semantic-cache"
		)
		return {
			"question":question,
			"answer":cached_result["answer"],
			"sources":cached_result["sources"],
			"model":"semantic-cache",
			"prompt_tokens":0,
			"completion_tokens":0,
			"total_tokens":0,
			"latency_ms":round(latency_ms,2),
			"estimated_cost_usd":0.0,
			"cache_hit":True,
			"cache_similarity":round(cached_result["score"],4),
			"cache_nli_score": cached_result["nli_score"]
		}
	#b.Cache-miss: run normal RAG

	#1:Retrieve relevant doc chunks and sources
	retrieved_docs=search_documents(question)
	sources=list(dict.fromkeys(
			doc["document_name"] for doc in retrieved_docs
		))

	#2.classify query and select model
	routing_decision=route_query(question)
	selected_model=routing_decision["model"]
	selected_route=routing_decision["route"]

	print("\nCost-Aware Routing")
	print("--------------------------------")
	print("Question:",question)
	print("Complexity:",routing_decision["complexity"])
	print("Reason:",routing_decision["reason"])
	print("Selected Model:",selected_model)
	print("Selected Route:",selected_route)

	#3.Generate answer using selected model

	generation=generate_answer(
		question,
		retrieved_docs,
		model_name=selected_model
	)
	answer=generation["answer"]

	#4.Calculate total request latency
	latency_ms=(perf_counter()-start_time)*1000

	#5.Estimate API cost using configured model rates.
	estimated_cost=estimate_cost(
		generation["prompt_tokens"],
		generation["completion_tokens"],
		model=generation["model"]
	)
	#6.cache save
	refusal_text="couldn't find this information"
	if (
		retrieved_docs
		and answer.strip()
		and refusal_text not in answer.lower()
	):
		save_to_cache(
			query=question,
			answer=answer,
			sources=sources
		)

	#7.Store request metrics in sqlite
	log_request(
		question=question,
		model=generation["model"],
		prompt_tokens=generation["prompt_tokens"],
		completion_tokens=generation["completion_tokens"],
		total_tokens=generation["total_tokens"],
		latency_ms=latency_ms,
		estimated_cost_usd=estimated_cost,
		cache_hit=False,
		route=selected_route
	)

	return {
		"question":question,
		"answer":answer,
		"sources":sources,
		"model":generation["model"],
		"prompt_tokens":generation["prompt_tokens"],
		"completion_tokens":generation["completion_tokens"],
		"total_tokens":generation["total_tokens"],
		"latency_ms":round(latency_ms,2),
		"estimated_cost_usd":estimated_cost,
		"cache_hit":False,
		"cache_similarity":None,
		"cache_nli_score":None,
		"route":selected_route,
		"complexity":routing_decision["complexity"],
		"routing_decision":routing_decision["reason"]
	}

if __name__ == "__main__":
	question="What is the annual sick leave entitlement?"
	result=ask_question(question)
	print("\nQuestion:",result["question"])
	print("\nAnswer:",result["answer"])
	print("\nSources:",result["sources"])
	print("\nModel:", result["model"])
	print("Route:",result.get("route", "semantic-cache"))
	print("Complexity:",result.get("complexity", "cached"))
	print("Cache hit:", result["cache_hit"])
	print("Cache similarity:", result["cache_similarity"])
	print("\nPrompt tokens:", result["prompt_tokens"])
	print("Completion tokens:", result["completion_tokens"])
	print("Total tokens:", result["total_tokens"])
	print("Latency (ms):", result["latency_ms"])
	print("Estimated cost (USD):", result["estimated_cost_usd"])