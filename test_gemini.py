
import os
import time
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("Testing Gemini 2.5 Flash...")
start = time.time()

try:
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents="Explain RAG in two short sentences.",
        config={"http_options": {"timeout": 30000}},
    )

    print("Response:", response.text)
    print(f"Time taken: {time.time() - start:.2f} seconds")

except Exception as error:
    print(f"Failed after {time.time() - start:.2f} seconds")
    print(type(error).__name__, error)
