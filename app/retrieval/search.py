
from qdrant_client import QdrantClient
from app.ingestion.model import embedding_model

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "rag_documents"

QUERY_PREFIX = (
    "Represent this sentence for searching relevant passages: "
)

client = QdrantClient(url=QDRANT_URL)


def search_documents(query: str, limit: int = 3):
    if not query.strip():
        raise ValueError("Search query cannot be empty")

    query_embedding = embedding_model.encode(
        QUERY_PREFIX + query,
        normalize_embeddings=True
    )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        limit=limit,
        with_payload=True
    ).points

    retrieved_documents = []

    for result in results:
        retrieved_documents.append({
            "text": result.payload["text"],
            "document_name": result.payload["document_name"],
            "source": result.payload["source"],
            "chunk_index": result.payload["chunk_index"],
            "score": result.score
        })

    return retrieved_documents


if __name__ == "__main__":
    question = "Can I transfer unused leave to next year?"
    documents = search_documents(question)

    print("\nQuestion:", question)
    print("\nRetrieved Documents:")

    for doc in documents:
        print("\n-------------------------------")
        print("Document:", doc["document_name"])
        print("Chunk:", doc["chunk_index"])
        print("Similarity score:", round(doc["score"], 4))
        print("Text:", doc["text"])
