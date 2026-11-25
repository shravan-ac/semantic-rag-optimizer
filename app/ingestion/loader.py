from pathlib import Path

BASE_DIR=Path(__file__).resolve().parents[2]
DATA_DIR= BASE_DIR /"data"/"raw"

def load_documents(folder_path: str):
	documents = []
	folder=Path(folder_path)
	supported_extensions={".txt",".md"}

	for file_path in folder.iterdir():
		if file_path.suffix.lower() not in supported_extensions:
			continue
		text=file_path.read_text(
			encoding="utf-8"
		)
		if not text.strip():
			continue
		documents.append({
			"document_name": file_path.name,
			"source":str(file_path),
			"text":text
		})

	return documents

if __name__=="__main__":
	docs=load_documents(DATA_DIR)
	print("Documents loaded:",len(docs))
	for doc in docs:
		print(doc["document_name"])