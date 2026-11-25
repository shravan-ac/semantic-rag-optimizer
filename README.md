# Semantic Caching & Cost-Aware Routing for RAG

An optimized Retrieval-Augmented Generation (RAG) system that reduces LLM inference costs and improves response efficiency through **semantic caching, intelligent model routing, and answer validation**.

## Overview

Traditional RAG systems send every user query through document retrieval and LLM generation, even when a similar question has already been answered.

This project introduces an optimization layer that:

- Uses semantic similarity to identify previously answered questions.
- Validates cached answers before reusing them.
- Routes queries to appropriate language models based on query complexity.
- Tracks token usage, estimated API costs, and latency.
- Evaluates answer quality and cache reliability.

The goal is to reduce unnecessary LLM calls while maintaining reliable, context-grounded answers.

## Architecture

```text
                  User Query
                      |
                      v
              Semantic Cache
                      |
             +--------+--------+
             |                 |
          Cache Hit         Cache Miss
             |                 |
             v                 v
       NLI Validation     Query Router
             |                 |
       +-----+-----+     +-----+------+
       |           |     |            |
    Valid       Invalid Simple      Complex
       |           |     |            |
       v           v     v            v
    Return      Retrieval + LLM Generation
    Cached          |
    Answer          v
                Cache Answer
                    |
                    v
               Final Answer
                    |
                    v
             Analytics Logging
```

## Tech Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Vector database | Qdrant |
| Embeddings | Sentence Transformers |
| Embedding model | BAAI/bge-small-en-v1.5 |
| Semantic validation | Cross-encoder NLI |
| LLM integration | LiteLLM |
| Language models | Gemini and OpenRouter models |
| Analytics storage | SQLite |
| Dashboard | Streamlit |
| Containerization | Docker |

## Key Features

### 1. Semantic Caching

Uses vector embeddings to retrieve semantically similar queries from Qdrant.

Unlike exact-match caching, it can identify differently worded questions with similar meanings.

### 2. NLI-Based Cache Validation

Uses `cross-encoder/nli-deberta-v3-small` to validate whether a cached answer is appropriate for a new query.

The validation process includes:

- Bidirectional natural language inference.
- Negation mismatch checks.
- Cache expiration checks.
- Corpus version validation.

This helps prevent incorrect cache reuse when questions have different meanings or constraints.

### 3. Cost-Aware Model Routing

The routing component classifies incoming queries and selects an appropriate model.

- Simple queries use a smaller, lower-cost model.
- Complex queries can be routed to a more capable model.

The objective is to avoid using expensive models for every request.

### 4. RAG Pipeline

The RAG pipeline performs:

1. Document loading.
2. Text chunking.
3. Embedding generation.
4. Vector storage in Qdrant.
5. Similarity-based retrieval.
6. Context-grounded answer generation.

### 5. Analytics Dashboard

The Streamlit dashboard provides visibility into:

- Cache hits and misses.
- Token consumption.
- Estimated LLM costs.
- Request latency.
- Model usage and routing.

## Project Structure

```text
semantic-rag-optimizer/
│
├── app/
│   ├── analytics/
│   │   ├── logger.py
│   │   └── pricing.py
│   │
│   ├── cache/
│   │   ├── semantic_cache.py
│   │   ├── semantic_validator.py
│   │   └── test_semantic_cache.py
│   │
│   ├── evaluation/
│   │   └── evaluation scripts and datasets
│   │
│   ├── generation/
│   │   └── generator.py
│   │
│   ├── ingestion/
│   │   ├── loader.py
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   ├── model.py
│   │   └── ingest.py
│   │
│   ├── retrieval/
│   │   └── search.py
│   │
│   ├── routing/
│   │   └── router.py
│   │
│   ├── rag.py
│   └── dashboard.py
│
├── data/
│   └── raw/
│       └── leave_policy.txt
│
├── evaluation/
│   ├── eval_dataset.json
│   ├── benchmark_report.md
│   └── benchmark result files
│
├── docker-compose.yml
├── requirements.txt
├── check_qdrant.py
└── .gitignore
```

## Getting Started

### Prerequisites

- Python 3.10
- Docker Desktop
- Gemini API key
- OpenRouter API key (if using an OpenRouter model)

### 1. Clone the repository

```bash
git clone https://github.com/shravan-ac/semantic-rag-optimizer.git

cd semantic-rag-optimizer
```

### 2. Create a virtual environment

```powershell
python -m venv .venv

.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root.

Add your own API credentials and model configuration:

```env
GEMINI_API_KEY=your_gemini_api_key
OPENROUTER_API_KEY=your_openrouter_api_key

SMALL_MODEL=your_small_model_identifier
LARGE_MODEL=your_large_model_identifier
```

Use model identifiers supported by your LiteLLM configuration. Keep API keys private and never commit `.env`.

### 5. Start Qdrant

```powershell
docker compose up -d
```

Qdrant will be available at:

```text
http://localhost:6333
```

### 6. Ingest documents

```powershell
python -m app.ingestion.ingest
```

### 7. Run the RAG application

```powershell
python -m app.rag
```

### 8. Launch the dashboard

```powershell
streamlit run app/dashboard.py
```

## Evaluation & Benchmark Results

The project was evaluated using a small leave-policy question-answering dataset containing 10 questions:

- 8 answerable questions.
- 2 unanswerable questions.

### Benchmark comparison

| Metric | Always-large baseline | Routed, no cache | Optimized, warm cache |
|---|---:|---:|---:|
| Average Token F1 | 0.6254 | 0.6962 | 0.8411 |
| Estimated cost | $0.0092637 | $0.0021759 | $0.0006405 |
| Average latency | 5611.26 ms | 3170.66 ms | 3055.66 ms |
| Total tokens | 6575 | 5273 | 1563 |
| Cache hit rate | — | — | 70% |

### Key observations

- The routed, no-cache configuration reduced estimated cost by approximately **76.51%** compared with the always-large baseline.
- The optimized warm-cache configuration reduced estimated cost by approximately **93.09%** compared with the always-large baseline.
- The warm-cache configuration achieved a **70% cache hit rate** on the evaluation dataset.
- Average Token F1 increased from 0.6254 in the always-large baseline to 0.8411 in the optimized configuration.

### Cache validation results

An isolated holdout evaluation used 20 query pairs:

| Metric | Result |
|---|---:|
| True Positives | 9 |
| False Positives | 3 |
| True Negatives | 7 |
| False Negatives | 1 |
| Precision | 0.75 |
| Recall | 0.90 |
| F1 Score | 0.8182 |

### Evaluation limitations

These results should be interpreted in context:

- The benchmark uses a small, domain-specific dataset.
- The optimized cache evaluation uses seven preloaded cache entries, so the 70% hit rate represents a warm-cache scenario rather than cold-start performance.
- All 10 benchmark queries were routed to the simple model; complex-model routing was not independently demonstrated by this dataset.
- Token F1 measures lexical overlap, not complete semantic correctness.
- Cost figures are estimates based on configured model pricing, not necessarily actual provider charges.

See [`evaluation/benchmark_report.md`](evaluation/benchmark_report.md) for additional benchmark details.

## Testing

Run the automated test suite:

```powershell
pytest -v
```

The tests cover:

- Exact query matches.
- Genuine paraphrases.
- Different topics.
- Different numerical constraints.
- Negation mismatches.
- Modality differences.

The cache tests use an isolated test collection to avoid modifying production cache data.

## Future Improvements

- Improve query complexity classification.
- Expand evaluation datasets across multiple domains.
- Improve cache validation for numerical and contextual constraints.
- Add more comprehensive semantic answer-quality evaluation.
- Evaluate cold-cache and long-term production performance.
- Explore adaptive routing based on latency, quality, and cost.

## Author

**Shravan A C**

GitHub: https://github.com/shravan-ac

---

*Built to explore practical techniques for making Retrieval-Augmented Generation systems more efficient, reliable, and cost-aware.*
