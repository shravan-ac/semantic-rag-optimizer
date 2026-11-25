
import os
from dotenv import load_dotenv

load_dotenv()


def estimate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model: str | None = None
) -> float | None:

    normalized_model = (model or "").lower()

    if "nemotron-3-ultra" in normalized_model:
        input_rate = os.getenv(
            "OPENROUTER_NEMOTRON_INPUT_USD_PER_MILLION"
        )
        output_rate = os.getenv(
            "OPENROUTER_NEMOTRON_OUTPUT_USD_PER_MILLION"
        )

    elif "flash-lite" in normalized_model:
        input_rate = os.getenv(
            "GEMINI_FLASH_LITE_INPUT_USD_PER_MILLION"
        )
        output_rate = os.getenv(
            "GEMINI_FLASH_LITE_OUTPUT_USD_PER_MILLION"
        )

    elif "flash" in normalized_model:
        input_rate = os.getenv(
            "GEMINI_FLASH_INPUT_USD_PER_MILLION"
        )
        output_rate = os.getenv(
            "GEMINI_FLASH_OUTPUT_USD_PER_MILLION"
        )

    else:
        input_rate = None
        output_rate = None

    # Legacy fallback applies only to Gemini models
    if "gemini" in normalized_model:
        if input_rate is None or output_rate is None:
            input_rate = os.getenv("GEMINI_INPUT_USD_PER_MILLION")
            output_rate = os.getenv("GEMINI_OUTPUT_USD_PER_MILLION")

    if input_rate is None or output_rate is None:
        return None

    input_rate = float(input_rate)
    output_rate = float(output_rate)

    input_cost = (prompt_tokens / 1_000_000) * input_rate
    output_cost = (completion_tokens / 1_000_000) * output_rate

    return input_cost + output_cost
