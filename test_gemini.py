import os
from dotenv import load_dotenv
from litellm import completion

load_dotenv()

response=completion(
						model=os.getenv("SMALL_MODEL"),
						messages=[
							{
								"role":"user",
								"content":"Explain RAG in two sentences."
							}
						],
						api_key=os.getenv("GEMINI_API_KEY")
						)
print(response.choices[0].message.content)
