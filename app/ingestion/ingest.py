import uuid
from pathlib import Path
from qdrant_client import QdrantClient,models
from app.ingestion.loader import load_documents
from app.ingestion.chunker import chunk_text
from app.ingestion.embedder import generate_embeddings

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "rag_documents"
CORPUS_VERSION = "v1"

BASE_DIR=Path(__file__).resolve().parents[2]
DATA_DIR=BASE_DIR/"data"/"raw"

def create_collection(client,vector_size):
	existing = [
		collection.name
		for collection in client.get_collections().collections
	]
	if COLLECTION_NAME not in existing:
		client.create_collection(
			collection_name=COLLECTION_NAME,
			vectors_config=models.VectorParams(
				size=vector_size,
				distance=models.Distance.COSINE
			)
		)
		print("Collection created!")
	else:
		print("Collection already exists.")

def ingest_documents():
	client=QdrantClient(url=QDRANT_URL)
	documents=load_documents(DATA_DIR)
	print("Documents found",len(documents))

	all_chunks=[]
	all_payloads=[]
	for doc in documents:
		chunks=chunk_text(doc["text"])
		for index,chunk in enumerate(chunks):
			all_chunks.append(chunk)
			all_payloads.append({
				"document_name":doc["document_name"],
				"source":doc["source"],
				"chunk_index":index,
				"text":chunk,
				"corpus_version":CORPUS_VERSION
			})
	if not all_chunks:
		print("No chunks found.")
		return
	embeddings=generate_embeddings(all_chunks)
	vector_size=embeddings.shape[1]
	create_collection(client,vector_size)
	points=[]

	for embedding,payload in zip(
		embeddings,
		all_payloads
	):
		point_id= str(uuid.uuid5(
			uuid.NAMESPACE_URL,
			f'{CORPUS_VERSION}:{payload["document_name"]}:{payload["chunk_index"]}'
		))
		points.append(
			models.PointStruct(
				id=point_id,
				vector=embedding.tolist(),
				payload=payload
			)
		)
	client.upsert(
		collection_name=COLLECTION_NAME,
		points=points,
		wait=True
	)
	print("\nIngestion Completed!")
	print("Documents:",len(documents))
	print("Chunks:",len(all_chunks))
	print("Vectors:",len(embeddings))

if __name__ == "__main__":
	ingest_documents()