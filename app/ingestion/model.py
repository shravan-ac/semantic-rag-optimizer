from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
embedding_model=SentenceTransformer(MODEL_NAME)
tokenizer=embedding_model.tokenizer
