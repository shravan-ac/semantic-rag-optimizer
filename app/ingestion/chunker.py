# from app.ingestion.model import tokenizer

# def chunk_text(text:str, chunk_size:int=450, overlap:int=80):
# 	if overlap>=chunk_size:
# 		raise ValueError("Overlap must be smaller than chunk size")
# 	tokens=tokenizer.encode(text,add_special_tokens=False)
# 	chunks=[]
# 	start=0
# 	while start < len(tokens):
# 		end=min(start+chunk_size, len(tokens))
# 		chunk_tokens=tokens[start:end]
# 		chunk=tokenizer.decode(chunk_tokens,skip_special_tokens=True)
# 		if chunk.strip():
# 			chunks.append(chunk)
# 		if end == len(tokens):
# 			break
# 		start = end-overlap
# 	return chunks

# if __name__ == "__main__":
# 	sample_text="""Employees are entitled to annual paid leave.
# 	Leave requests must be submitted 5 days in advance.Managers approve leave requests."""

# 	result= chunk_text(sample_text)
# 	for i,chunk in enumerate(result):
# 		print(f"\nChunk{i+1}:")
# 		print(chunk)


#SECTION BASED CHUNKING -preserve logical meaning

import re


def chunk_text(text: str):
    # Recognize policy headings even when they appear
    # in the middle of a line.
    pattern = (
        r"(?i)(?=\b(?:"
        r"annual leave|"
        r"leave application|"
        r"approval|"
        r"carry forward|"
        r"sick leave|"
        r"medical documentation|"
        r"emergency leave"
        r")\s*:)"
    )

    sections = re.split(pattern, text)

    chunks = []

    for section in sections:
        section = section.strip()

        if section:
            chunks.append(section)

    return chunks
