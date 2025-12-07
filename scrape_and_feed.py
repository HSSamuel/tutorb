import os
import requests
import time # Add this import at the top
from bs4 import BeautifulSoup
from pypdf import PdfReader
from dotenv import load_dotenv
# 👇 CHANGED: Import Cohere to match your new Backend
from langchain_cohere import CohereEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import create_client, Client

# 1. Setup & Config
load_dotenv()
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
cohere_key = os.environ.get("COHERE_API_KEY")

supabase: Client = create_client(supabase_url, supabase_key)

print("⏳ Connecting to Cloud Embeddings (Cohere)...")
# 👇 CHANGED: We use the cloud model to save memory and match the backend
# 'embed-english-light-v3.0' is 384 dimensions, fitting your database perfectly.
embeddings = CohereEmbeddings(
    model="embed-english-light-v3.0", 
    cohere_api_key=cohere_key
)

# --- HELPER FUNCTIONS ---

def get_text_from_url(url):
    """Downloads and extracts text from a Website"""
    print(f"🌐 Fetching URL: {url}...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Get all paragraphs
        paragraphs = soup.find_all('p')
        text = "\n\n".join([p.get_text() for p in paragraphs])
        return text
    except Exception as e:
        print(f"❌ Failed to scrape URL: {e}")
        return None

def get_text_from_pdf(pdf_path):
    """Extracts text from a local PDF file"""
    print(f"📄 Reading PDF: {pdf_path}...")
    try:
        reader = PdfReader(pdf_path)
        text = ""
        # Loop through every page
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"❌ Failed to read PDF: {e}")
        return None

def process_and_upload(text, source_name, region_tag="Global"):
    """Chunks text and uploads to Supabase using BATCHING to save API credits"""
    if not text or len(text) < 100:
        print("   ⚠️ Text too short or empty. Skipping.")
        return

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = text_splitter.split_text(text)
    
    print(f"   🔪 Split into {len(chunks)} chunks. Generating Vectors...")

    # --- BATCHING LOGIC STARTS HERE ---
    
    # 1. Generate Vectors in Batches
    all_vectors = []
    
    # This sends less text per call, keeping you under the 100k token limit.
    batch_size = 20  
    
    try:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            print(f"   ⚡ Embedding batch {i} to {i+len(batch)}...")
            
            # embed_documents sends the whole list in ONE call
            batch_vectors = embeddings.embed_documents(batch)
            all_vectors.extend(batch_vectors)
            
            # 👇 CHANGE 2: Increase sleep from 1 to 10 seconds
            # This forces the script to wait, allowing your "minute quota" to reset.
            time.sleep(10)
            
    except Exception as e:
        print(f"   ❌ Error generating vectors: {e}")
        return

    # 2. Upload to Supabase (Now we have all vectors ready)
    print(f"   💾 Uploading {len(chunks)} entries to Database...")
    
    for i, chunk in enumerate(chunks):
        final_content = f"{chunk}\n(Source: {source_name})"
        
        supabase.table("cultural_knowledge").insert({
            "content": final_content,
            "region": region_tag,
            "embedding": all_vectors[i]
        }).execute()
        
    print(f"   ✅ Finished uploading {source_name}!")

# --- MAIN EXECUTION LOOP ---
if __name__ == "__main__":
    source_file = "sources.txt"
    
    if not os.path.exists(source_file):
        print(f"❌ Error: Could not find {source_file}. Please create it first.")
    else:
        print(f"🚀 Starting Mega-Ingestion from {source_file}...")
        
        with open(source_file, "r") as f:
            lines = f.readlines()

        for line in lines:
            source = line.strip()
            if not source: continue # Skip empty lines

            raw_text = None
            
            # ROUTER LOGIC: Check if it's a URL or PDF
            if source.startswith("http"):
                raw_text = get_text_from_url(source)
            elif source.endswith(".pdf"):
                # Assume it's a file in the same folder
                raw_text = get_text_from_pdf(source)
            else:
                print(f"❓ Unknown source type: {source}")

            # Upload if we got text
            if raw_text:
                process_and_upload(raw_text, source, "General Knowledge")
        
        print("\n🎉 ALL SOURCES PROCESSED!")