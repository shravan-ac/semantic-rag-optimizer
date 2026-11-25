from app.ingestion.model import embedding_model


def generate_embeddings(texts: list[str]):
	embeddings=embedding_model.encode(
		texts,
		normalize_embeddings=True,
		show_progress_bar=True
	)
	return embeddings

if __name__ == "__main__":
	sample_texts=[
		"How do I reset my password?",
		"What are the steps for password recovery?",
		"What is the company's refund policy?"
	]
	vectors=generate_embeddings(sample_texts)
	print("Number of vectors:",len(vectors))
	print("Vector dimensions:",vectors.shape)
	print("First vector:",vectors[0][:10])

# similarities=util.cos_sim(vectors,vectors)
# print(similarities)