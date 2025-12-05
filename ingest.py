import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from supabase.client import Client, create_client

# 1. Load Env Vars
load_dotenv()
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY") 

# 2. Connect to Supabase
supabase: Client = create_client(supabase_url, supabase_key)

# 3. Define our Data
raw_data = [
    {
        "text": "In Lagos traffic (Go-slow), danfo drivers often cut corners to move faster. This is like electricity taking the path of least resistance.",
        "region": "Lagos, Nigeria"
    },
    {
        "text": "A market in Yaba is chaotic but organized; prices are determined by bargaining power. This explains supply and demand dynamics perfectly.",
        "region": "Lagos, Nigeria"
    },
    {
        "text": "The Matatu culture in Nairobi involves loud music and vibrant art to attract customers. This is similar to how flowers use colors to attract pollinators.",
        "region": "Nairobi, Kenya"
    }
]

# 4. Generate Embeddings (FREE LOCAL MODEL)
print("⏳ Loading local AI model (this might take 10 seconds)...")
# This downloads a small free model (all-MiniLM-L6-v2) to your computer
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("🚀 Generating embeddings and uploading...")

for item in raw_data:
    # We embed the text locally
    vector = embeddings.embed_query(item["text"])
    
    # We insert into Supabase
    data, count = supabase.table("cultural_knowledge").insert({
        "content": item["text"],
        "region": item["region"],
        "embedding": vector
    }).execute()

print("✅ Success! Database populated with local metaphors (Free Version).")