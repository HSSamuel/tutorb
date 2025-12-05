from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

print("Testing OpenAI Key...")

try:
    embeddings = OpenAIEmbeddings()
    # Try to embed a single word. If this works, your key is perfect.
    vector = embeddings.embed_query("Hello")
    print("✅ SUCCESS! Your OpenAI key is working.")
except Exception as e:
    print(f"❌ ERROR: {e}")