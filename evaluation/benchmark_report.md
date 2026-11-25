# Semantic Caching and Cost-Aware Routing for RAG

## 1. Project overview

This project implements a Retrieval-Augmented Generation (RAG) system
that aims to reduce inference cost and response latency while preserving
answer quality.

The system combines:

-   Document ingestion, chunking, and embedding generation.
-   Vector retrieval using Qdrant.
-   Semantic caching using BGE embeddings.
-   Natural Language Inference (NLI) validation before reusing cached
    answers.
-   Cost-aware model routing between a simpler model and a more capable
    model.
-   Answer generation through LiteLLM.
-   Evaluation of retrieval, cache decisions, answer quality, latency,
    and estimated generation cost.

The evaluation corpus used in the recorded experiments is an employee
leave-policy document.

## 2. System workflow

For an incoming question, the optimized RAG flow is:

1.  Check the semantic cache for a potentially equivalent question.
2.  Retrieve candidate cache entries using vector similarity.
3.  Validate candidate question equivalence using bidirectional NLI
    scoring and additional safeguards.
4.  If a candidate passes validation, return its cached answer and
    sources.
5.  Otherwise, retrieve relevant document chunks from Qdrant.
6.  Route the question to a model using the configured routing rules.
7.  Generate an answer grounded in the retrieved evidence.
8.  Save supported answers to the semantic cache for possible reuse.

The benchmark runner used a separate temporary Qdrant collection. It did
not use the production cache collection or write to SQLite analytics.
The temporary collection was deleted after the benchmark.

## 3. Evaluation methodology

The project evaluated several distinct capabilities. Their results
should not be treated as interchangeable:

-   **Retrieval evaluation:** checks whether expected evidence chunks
    appear among the top retrieved results.
-   **Cache validation evaluation:** measures whether semantically
    equivalent questions are accepted and non-equivalent questions are
    rejected.
-   **Generation evaluation:** measures model output, estimated cost,
    and latency without semantic-cache reuse.
-   **Full optimization evaluation:** measures the behavior of the
    warm-cache flow, including cache hits and generation on misses.
-   **Answer-quality evaluation:** compares generated or cached answers
    with reference answers using lexical token overlap and checks
    refusals for unanswerable questions.

### Dataset

The RAG evaluation dataset contains 10 questions:

-   8 answerable questions about the leave policy.
-   2 unanswerable questions about maternity leave and dental insurance.
-   One question asks whether 10 annual leave days can be carried
    forward, testing whether the system preserves the policy's five-day
    limit.

The cache holdout validation set contains 20 question pairs:

-   10 equivalent pairs.
-   10 non-equivalent pairs.

## 4. Retrieval evaluation

The initial retrieval evaluation found the expected evidence chunk in
the top three results for all 8 answerable test cases.

  Metric                             Result
  -------------------------------- --------
  Answerable test cases                   8
  Expected chunks found in top 3          8
  Hit Rate@3                           100%

This is a result on a small, policy-specific dataset. It should not be
interpreted as a general retrieval-performance guarantee. Similarity
scores alone also do not establish that a document contains evidence for
an unanswerable question.

## 5. Semantic-cache validation

The semantic cache uses BGE embeddings to retrieve candidate questions
and a bidirectional NLI score to validate whether a candidate question
is sufficiently equivalent to the new question.

The recorded isolated holdout evaluation used a threshold of 0.27458.

  Metric                   Result
  ---------------------- --------
  Total question pairs         20
  True positives                9
  False positives               3
  True negatives                7
  False negatives               1
  Precision                  0.75
  Recall                     0.90
  F1                       0.8182

The three false positives included distinctions involving working days
versus calendar days, requesting emergency leave versus automatic
approval, and annual entitlement versus carry-forward allowance. One
false negative involved a paraphrase of the sick-leave allowance
question.

These errors show that NLI validation is useful but not a guarantee of
question equivalence. The current results justify continued testing,
particularly around quantities, conditions, permissions, and policy
scope.

The isolated holdout test used temporary collections and did not clear
or replace the production cache.

## 6. Baseline and routed-generation comparison

Two no-cache generation configurations were evaluated on the same
10-question dataset:

-   **Always-large baseline:** every question was sent to the configured
    complex model.
-   **Routed generation:** the configured router selected a model for
    each question. In this particular run, all 10 questions were
    classified as simple and used Gemini Flash Lite.

  Metric                              Always-large baseline   Routed generation, no cache
  --------------------------------- ----------------------- -----------------------------
  Successful generations                              10/10                         10/10
  Total tokens                                        6,575                         5,273
  Average latency                               5,611.26 ms                   3,170.66 ms
  Estimated generation cost                     \$0.0092637                   \$0.0021759
  Average answerable Token F1                        0.6254                        0.6962
  Unanswerable refusal detections                       2/2                           2/2

For this run, routed generation reduced estimated cost by approximately
76.5% and average latency by approximately 43.5% compared with the
always-large baseline. Its average answerable Token F1 was higher by
0.0708.

Important qualification: all 10 questions in this dataset were routed to
the simple model. This run therefore does not demonstrate the quality or
cost behavior of the complex-model route on a representative set of
complex questions.

## 7. Full optimization benchmark

The full optimization benchmark evaluated all 10 questions with a warm
cache. Seven curated, policy-grounded question-answer entries were
preloaded into an isolated experiment collection before the questions
were tested.

  Metric                                         Result
  -------------------------------------- --------------
  Questions                                          10
  Successful                                         10
  Errors                                              0
  Cache hits                                          7
  Cache misses                                        3
  Warm-cache hit rate                             70.0%
  Generated tokens                                1,563
  Questions with known generation cost                3
  Estimated generation cost                 \$0.0006405
  Average latency                           3,055.66 ms
  p50 latency                               1,069.19 ms
  p95 latency                              10,229.16 ms

The seven cache hits avoided generation. The three misses used the
generation path. The misses included the advance-notice question and the
two unanswerable questions.

The benchmark's 70% hit rate is a **warm-cache result** on a small,
fixed dataset. Because the cache was manually preloaded with curated
answers, this is not a cold-start result, a measured production hit
rate, or evidence of expected hit rate under live traffic.

The benchmark used an isolated temporary collection, did not use the
production cache, did not write to SQLite analytics, and deleted the
temporary collection after the run.

## 8. Full optimization answer quality

The answer-quality evaluator compared the 10 optimized answers with the
dataset's reference answers.

  Metric                             Result
  -------------------------------- --------
  Questions evaluated                    10
  Missing results                         0
  Generation errors                       0
  Answerable questions scored             8
  Average answerable Token F1        0.8411
  Unanswerable questions checked          2
  Correct refusal detections            2/2
  Refusal detection rate               100%
  Cache-hit answerable Token F1      0.8584
  Cache-miss answerable Token F1     0.7200

The cache-miss answerable Token F1 is based on only one answerable
question; the other two misses were unanswerable questions. Therefore,
the hit-versus-miss F1 difference should not be interpreted as a
reliable comparison of answer quality.

Manual review of all 10 answer rows found the answers factually
consistent with the available leave-policy evidence. The answer to the
10-day carry-forward question stated the five-day maximum, and both
unanswerable questions received refusals.

### Token F1 limitation

Token precision, recall, and F1 measure lexical overlap after
tokenization and stopword removal. They do not establish semantic
equivalence, factual correctness, completeness, or policy safety. For
example, a factually correct paraphrase can receive a lower score than a
reference answer with more matching words.

The refusal evaluator uses phrase matching. Its 2/2 result is useful as
a basic check, but phrase matching is not a complete measure of refusal
correctness.

## 9. Three-configuration summary

  -----------------------------------------------------------------------
  Metric               Always-large   Routed, no cache    Optimized, warm
                           baseline                                 cache
  -------------- ------------------ ------------------ ------------------
  Average                    0.6254             0.6962             0.8411
  answerable
  Token F1                                             

  Estimated             \$0.0092637        \$0.0021759        \$0.0006405
  generation                                           
  cost                                                 

  Average               5,611.26 ms        3,170.66 ms        3,055.66 ms
  latency                                              

  Cache hit rate     Not applicable     Not applicable              70.0%
  -----------------------------------------------------------------------

Compared with routed generation without caching, the warm-cache
benchmark's estimated generation cost was approximately 70.6% lower. Its
average latency was approximately 3.6% lower.

Compared with the always-large baseline, its estimated generation cost
was approximately 93.1% lower.

These are observed results from this small experiment, not controlled
production guarantees. The answer-quality comparison also has
limitations: the optimized configuration uses curated warm-cache
entries, and model routing selected the simple model for every question
in the recorded routed-generation run.

### Cost interpretation

The cost values are estimates based on the project's configured
model-pricing assumptions and recorded token usage. They are not
verified provider invoices or actual charges. The complex baseline used
a `:free` provider endpoint while the project pricing estimator applied
configured paid-tier-equivalent estimates. Interpret the baseline cost
as a modeled comparison, not an amount actually billed.

## 10. Limitations

1.  **Small dataset:** The main RAG evaluation contains only 10
    questions from one leave-policy document.
2.  **Warm-cache setup:** Seven manually curated answers were inserted
    before the optimization benchmark. The 70% hit rate is not a
    cold-cache or production measurement.
3.  **NLI false positives:** The 20-pair holdout test produced three
    false positives, including policy distinctions that matter.
4.  **Limited routing coverage:** The recorded routed-generation dataset
    sent every question to the simple model. Complex-route quality needs
    a separate representative evaluation.
5.  **Lexical quality metric:** Token F1 cannot independently verify
    meaning or factual correctness.
6.  **Refusal metric:** Phrase-based refusal detection can miss valid
    refusals or flag wording incorrectly.
7.  **Latency variability:** The small number of requests and occasional
    slow generation calls make latency estimates sensitive to individual
    outliers.
8.  **Estimated cost:** Pricing estimates are not confirmed provider
    charges.

## 11. Future improvements

-   Expand the evaluation corpus with more documents, topics, and
    question types.
-   Add a representative set of genuinely complex questions to evaluate
    both routing paths.
-   Test cache behavior with cold-start, warm-cache, and simulated
    repeated-traffic workloads.
-   Expand cache validation pairs around numbers, units, dates,
    negation, permissions, exceptions, and policy conditions.
-   Review false-positive cache decisions and improve validation without
    relying on dataset-specific hardcoded rules.
-   Add semantic or model-assisted answer evaluation alongside manual
    factual review.
-   Repeat latency benchmarks with multiple runs and report
    distributions.
-   Compare estimated cost assumptions with actual provider usage where
    available.
-   Test cache expiration, corpus-version invalidation, and source
    freshness.

## 12. Conclusion

The experiments demonstrate a working RAG prototype that combines vector
retrieval, semantic caching, NLI-based cache validation, model routing,
and answer generation.

On the recorded 10-question warm-cache benchmark, the system achieved a
70% cache hit rate, 0.8411 average answerable Token F1, and an estimated
generation cost of \$0.0006405. The routed no-cache run and always-large
baseline provide useful comparison points.

The results are promising for this small policy dataset. Further
evaluation on larger and more diverse data, especially complex routing
and cache false-positive cases, is necessary before making broader
claims about quality, savings, or production readiness.

------------------------------------------------------------------------

## Reproducibility artifacts

The recorded evaluation outputs include:

-   `evaluation/eval_dataset.json`
-   `evaluation/generation_only_results.json`
-   `evaluation/answer_quality_summary.json`
-   `evaluation/baseline_results.json`
-   `evaluation/baseline_answer_quality_summary.json`
-   `evaluation/cache_holdout_isolated_results.json`
-   `evaluation/full_optimization_results.json`
-   `evaluation/full_optimization_quality_results.csv`
-   `evaluation/full_optimization_quality_summary.json`

The figures in this report reflect the experiment outputs recorded
during development. Re-run the evaluation scripts after changing models,
pricing assumptions, corpus content, cache thresholds, or routing rules.
