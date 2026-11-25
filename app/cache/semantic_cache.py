
import time
import uuid

from qdrant_client import QdrantClient, models

from app.ingestion.model import embedding_model

from app.cache.semantic_validator import (
    score_question_pairs_bidirectional,
    NLI_MEAN_THRESHOLD,
    has_explicit_negation,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

QDRANT_URL = "http://localhost:6333"

CACHE_COLLECTION = "semantic_cache_v2"

VECTOR_SIZE = 384

CACHE_TTL_SECONDS = 24 * 60 * 60

CORPUS_VERSION = "v1"

CANDIDATE_LIMIT = 10


client = QdrantClient(url=QDRANT_URL)


# --------------------------------------------------
# Create cache collection
# --------------------------------------------------

def create_cache_collection(collection_name: str = CACHE_COLLECTION):
    collections = client.get_collections().collections
    existing = [collection.name for collection in collections]

    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE
            )
        )
        print(f"Cache collection '{collection_name}' created.")
    else:
        print(f"Cache collection '{collection_name}' already exists.")

# --------------------------------------------------
# Generate cache embeddings
# --------------------------------------------------

def embed_query(query: str):

    embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    )

    return embedding.tolist()


# --------------------------------------------------
# Search semantic cache
# --------------------------------------------------

def search_cache(query, collection_name: str = CACHE_COLLECTION):

    if not query or not query.strip():
        return None

    vector = embed_query(query)

    current_time = time.time()

    results = client.query_points(
        collection_name=collection_name,
        query=vector,
        limit=CANDIDATE_LIMIT,
        with_payload=True
    ).points

    valid_candidates = []

    # First filter candidates using metadata.
    for result in results:

        payload = result.payload or {}

        if payload.get("corpus_version") != CORPUS_VERSION:
            continue

        if payload.get("expires_at", 0) <= current_time:
            continue

        cached_query = payload.get("query")

        if not cached_query:
            continue

        if not payload.get("answer") or not payload.get("sources"):
            continue

        valid_candidates.append({
            "cached_query": cached_query,
            "answer": payload["answer"],
            "sources": payload["sources"],
            "vector_score": float(result.score)
        })

    if not valid_candidates:

        print("No valid cache candidates found.")

        return None

    # --------------------------------------------------
    # Explicit negation check
    # --------------------------------------------------

    candidates_for_nli = []

    incoming_has_negation = has_explicit_negation(query)

    for candidate in valid_candidates:

        cached_has_negation = has_explicit_negation(
            candidate["cached_query"]
        )

        if incoming_has_negation != cached_has_negation:

            print(
                "Cache rejected: explicit negation mismatch:",
                candidate["cached_query"]
            )

            continue

        candidates_for_nli.append(candidate)

    if not candidates_for_nli:

        print("No candidates passed the negation check.")

        return None

    # --------------------------------------------------
    # NLI mean validation
    # --------------------------------------------------

    question_pairs = [
        (query, candidate["cached_query"])
        for candidate in candidates_for_nli
    ]

    nli_scores = score_question_pairs_bidirectional(
        question_pairs
    )

    accepted_candidates = []

    for candidate, nli_score in zip(
        candidates_for_nli,
        nli_scores
    ):

        print(
            f"Cache candidate: {candidate['cached_query']} "
            f"| BGE similarity: {candidate['vector_score']:.4f} "
            f"| NLI mean: {nli_score:.5f}"
        )

        if nli_score >= NLI_MEAN_THRESHOLD:

            accepted_candidates.append({
                **candidate,
                "nli_score": float(nli_score)
            })

    if not accepted_candidates:

        print("Cache rejected: no candidate passed NLI validation.")

        return None

    # --------------------------------------------------
    # Select the best candidate
    # --------------------------------------------------

    best_candidate = max(
        accepted_candidates,
        key=lambda item: item["nli_score"]
    )

    print(
        f"Cache HIT: NLI mean = "
        f"{best_candidate['nli_score']:.5f}"
    )

    return {
        "answer": best_candidate["answer"],
        "sources": best_candidate["sources"],
        "score": best_candidate["vector_score"],
        "nli_score": best_candidate["nli_score"]
    }


# --------------------------------------------------
# Save answer to semantic cache
# --------------------------------------------------

def save_to_cache(
    query: str,
    answer: str,
    sources: list[str],
    collection_name: str = CACHE_COLLECTION
):

    if (
        not query
        or not query.strip()
        or not answer
        or not answer.strip()
        or not sources
    ):
        return

    vector = embed_query(query)

    current_time = time.time()

    payload = {
        "query": query,
        "answer": answer,
        "sources": sources,
        "corpus_version": CORPUS_VERSION,
        "created_at": current_time,
        "expires_at": current_time + CACHE_TTL_SECONDS
    }

    client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload
            )
        ]
    )

    print("Answer saved to semantic cache.")
