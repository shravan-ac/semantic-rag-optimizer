
import os

from dotenv import load_dotenv
from litellm import completion

load_dotenv()

DEFAULT_MODEL = os.getenv(
    "GEMINI_SIMPLE_MODEL",
    "gemini/gemini-flash-lite-latest"
)

NO_INFORMATION_MESSAGE = (
    "I couldn't find this information in the available documents."
)


def generate_answer(
    question: str,
    retrieved_docs: list[dict],
    model_name: str | None = None
) -> dict:

    selected_model = model_name or DEFAULT_MODEL

    # Check the API key required by the selected provider
    normalized_model = selected_model.lower()

    if normalized_model.startswith("openrouter/"):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not found. Check your .env file."
            )

    elif normalized_model.startswith("gemini/"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not found. Check your .env file."
            )

    # Handle empty retrieval results
    if not retrieved_docs:
        return {
            "answer": NO_INFORMATION_MESSAGE,
            "model": selected_model,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

    # Build context from retrieved documents
    context_parts = []

    for index, doc in enumerate(retrieved_docs, start=1):

        source = doc.get("document_name", "Unknown source")
        text = doc.get("text", "")

        if not text.strip():
            continue

        context_parts.append(
            f"""
Document {index}
Source: {source}
Content:
{text}
"""
        )

    context = "\n\n".join(context_parts)

    if not context.strip():
        return {
            "answer": NO_INFORMATION_MESSAGE,
            "model": selected_model,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

    # Grounded system prompt
    system_prompt = """
You are a careful assistant answering questions using company documents.

Your answers must be grounded strictly in the provided document context.

Follow these rules:

1. Use only information explicitly supported by the provided documents.

2. Never invent policy rules, exceptions, dates, numbers,
   requirements, or approval conditions.

3. Do not use general knowledge or assumptions to fill gaps
   in the documents.

4. If a detail is not explicitly mentioned, clearly state:
   "The available documents do not specify this."

5. Do not infer exceptions from the purpose or wording of a policy.
   For example, if emergency leave is mentioned but an exception
   to the general advance-notice rule is not explicitly stated,
   do not claim that such an exception exists.

6. Distinguish between:
   - Explicitly stated information.
   - Information not specified in the documents.
   - Inferences that cannot be confirmed from the documents.

7. When comparing policies, compare only the details supported
   by the context. Do not assume that missing details are identical.

8. If documents contain conflicting information, identify
   the conflict instead of choosing one version.

9. Mention the source document when presenting relevant facts.

10. Be clear, concise, and directly answer the question.

11. If the documents do not contain enough information to answer
    a question, explain what information is missing rather than
    making up an answer.
"""

    user_prompt = f"""
DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

INSTRUCTIONS:

Answer using only the document context.

For comparisons or evaluations:
- Explain the facts explicitly supported by the documents.
- Identify important information that is not specified.
- Never present an unsupported inference as an established policy rule.
- Cite the relevant source document.

ANSWER:
"""

    # Generate answer using the model selected by the router
    response = completion(
        model=selected_model,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        max_tokens=800,
        timeout=60,
        num_retries=0
    )

    # Extract token usage safely
    usage = response.usage

    prompt_tokens = (
        getattr(usage, "prompt_tokens", 0) or 0
        if usage else 0
    )

    completion_tokens = (
        getattr(usage, "completion_tokens", 0) or 0
        if usage else 0
    )

    total_tokens = (
        getattr(usage, "total_tokens", 0) or
        (prompt_tokens + completion_tokens)
        if usage else
        prompt_tokens + completion_tokens
    )

    # Extract generated answer
    answer = response.choices[0].message.content

    return {
        "answer": answer or "I couldn't generate an answer.",
        "model": selected_model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens
    }
