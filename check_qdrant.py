from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")

collections = client.get_collections()

print("Successfully connected to Qdrant!")
print("Existing collections:", collections.collections)
