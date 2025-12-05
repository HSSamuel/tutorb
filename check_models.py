import os
import cohere
from dotenv import load_dotenv

# 1. Load your key
load_dotenv()
api_key = os.environ.get("COHERE_API_KEY")

if not api_key:
    print("❌ Error: No API Key found in .env")
else:
    print(f"✅ Key found: {api_key[:5]}...")

# 2. Connect to Cohere directly
try:
    client = cohere.Client(api_key)
    response = client.models.list()
    
    print("\n📋 AVAILABLE MODELS FOR YOU:")
    print("-----------------------------")
    # Loop through and print only the 'chat' compatible models
    found_any = False
    for model in response.models:
        if "chat" in model.endpoints:
            print(f"• {model.name}")
            found_any = True
            
    if not found_any:
        print("(No specific chat models found, printing ALL names):")
        for model in response.models:
            print(f"• {model.name}")

except Exception as e:
    print(f"\n❌ FAILED to list models: {e}")