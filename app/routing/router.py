import os
from dotenv import load_dotenv

load_dotenv()


#MODEL_CONFIGURATION

SIMPLE_MODEL=os.getenv(
	"GEMINI_SIMPLE_MODEL",
	"gemini/gemini-flash-lite-latest"
)

COMPLEX_MODEL=os.getenv(
	"OPEN_ROUTER_MODEL",
	"openrouter/nvidia/nemotron-3-ultra-550b-a55b:free"
)

#COMPLEXITY INDICATORS

COMPLEXITY_KEYWORDS={
    "compare",
    "comparison",
    "analyze",
    "analysis",
    "evaluate",
    "explain",
    "difference",
    "differences",
    "relationship",
    "relationships",
    "summarize",
    "summary",
    "recommend",
    "recommendation",
    "advantages",
    "disadvantages",
    "pros",
    "cons",
    "multiple",
    "impact",
    "implications",
    "justify",
    "reasoning",
    "step-by-step",
    "calculate",
}

#RULE BASED QUERY CLASSIFICATION
def classify_query(question:str) -> dict:
	"""
	Classify a query as simple or somplex using predefined rules.
	No LLM API call is required for classification.
	"""
	if not question or not question.strip():
		raise ValueError("Question cannot be empty.")

	normalized_question=question.lower().strip()
	words=normalized_question.split()
	word_count=len(words)

	matched_keywords=[
		keyword
		for keyword in COMPLEXITY_KEYWORDS
		if keyword in normalized_question
	]

	#Rule 1: long qns are considered complex.
	if word_count>25:
		return{
			"complexity":"complex",
			"reason":"Question contains more than 25 words."
		}

	#Rule 2: qns containinng complex indicators are considered complex.
	if matched_keywords:
		return {
			"complexity":"complex",
			"reason":(
				"Complexity keywords detected: "
				+",".join(sorted(matched_keywords))
			)
		}

	#Rule 3: Remaining questions use the simpler model.
	return {
		"complexity":"simple",
		"reason":"No complexity indicators detected."
	}

#MODEL ROUTING

def route_query(question:str) -> dict:
	"""
	Select a Gemini model based on query complexity
	"""
	classification= classify_query(question)
	if classification["complexity"] == "complex":
		return {
			"model":COMPLEX_MODEL,
			"route":"rule-complex",
			"complexity":"complex",
			"reason":classification["reason"]
		}

	return {
		"model":SIMPLE_MODEL,
		"route":"rule-simple",
		"complexity":"simple",
		"reason":classification["reason"]
	}
