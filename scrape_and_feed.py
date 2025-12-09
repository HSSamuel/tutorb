import os
import requests
import time 
import string
from bs4 import BeautifulSoup
from pypdf import PdfReader
from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import create_client, Client

# 1. Setup & Config
load_dotenv()
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
cohere_key = os.environ.get("COHERE_API_KEY")

if not all([supabase_url, supabase_key, cohere_key]):
    raise ValueError("❌ Missing API Keys in .env file")

supabase: Client = create_client(supabase_url, supabase_key)

print("⏳ Connecting to Cloud Embeddings (Cohere)...")
embeddings = CohereEmbeddings(
    model="embed-english-light-v3.0", 
    cohere_api_key=cohere_key
)

# --- HELPER FUNCTIONS ---

def is_garbage_text(text):
    """Returns True if the text looks like corrupted characters"""
    if not text: return True
    
    # Calculate percentage of readable characters
    printable = set(string.printable)
    clean_chars = sum(1 for c in text if c in printable)
    total_chars = len(text)
    
    if total_chars == 0: return True
    
    ratio = clean_chars / total_chars
    return ratio < 0.8

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
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
        return text
    except Exception as e:
        print(f"❌ Failed to read PDF: {e}")
        return None

def process_and_upload(text, source_name, region_tag="Global"):
    """Chunks text and uploads to Supabase using SMART RATE LIMITING"""
    if not text or len(text) < 100:
        print(f"   ⚠️ Text too short in {source_name}. Skipping.")
        return
        
    if is_garbage_text(text):
        print(f"   🛑 SKIPPING {source_name}: Text looks corrupted/encrypted.")
        return

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = text_splitter.split_text(text)
    
    print(f"   🔪 Split into {len(chunks)} chunks. Generating Vectors...")

    # --- SMART BATCHING LOGIC ---
    all_vectors = []
    batch_size = 5  # Reduced size to stay under limits
    
    i = 0
    while i < len(chunks):
        batch = chunks[i : i + batch_size]
        print(f"   ⚡ Embedding batch {i} to {i+len(batch)}...")
        
        try:
            # Generate vectors
            batch_vectors = embeddings.embed_documents(batch)
            all_vectors.extend(batch_vectors)
            
            # If successful, move to next batch
            i += batch_size
            
            # Gentle wait to be kind to the API
            time.sleep(2) 
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "TooManyRequests" in error_msg or "quota" in error_msg.lower():
                print("   ⏳ Rate Limit Hit! Sleeping for 65 seconds to reset quota...")
                time.sleep(65)
                # We do NOT increment 'i', so it retries the same batch
            else:
                print(f"   ❌ Critical Error: {e}")
                return # Stop if it's a non-recoverable error

    # 2. Upload to Supabase
    print(f"   💾 Uploading {len(chunks)} entries to Database...")
    
    for i, chunk in enumerate(chunks):
        final_content = f"{chunk}\n(Source: {source_name})"
        
        try:
            supabase.table("cultural_knowledge").insert({
                "content": final_content,
                "region": region_tag,
                "embedding": all_vectors[i]
            }).execute()
        except Exception as e:
            # Sometimes specific chunks fail, just log and continue
            print(f"   ⚠️ DB Insert Warning: {e}")

    print(f"   ✅ Finished uploading {source_name}!")

# --- MAIN EXECUTION LOOP ---
if __name__ == "__main__":
    print(f"🚀 Starting Mega-Ingestion...")

    # 1. Process URLs from sources.txt
    if os.path.exists("sources.txt"):
        print("\n--- 🌐 Processing URLs from sources.txt ---")
        with open("sources.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                url = line.strip()
                if url and url.startswith("http"):
                    raw_text = get_text_from_url(url)
                    if raw_text:
                        process_and_upload(raw_text, url, "General Knowledge")

    # 2. Process PDFs
    print("\n--- 📄 Scanning for PDFs in folder ---")
    files = [f for f in os.listdir('.') if f.endswith('.pdf')]
    
    # Optional: If you want to retry ONLY the failed grammar book, uncomment below:
    # files = ["Advanced-Learners-Grammar.pdf"] 
    
    if not files:
        print("No PDF files found in this folder.")
    
    for filename in files:
        raw_text = get_text_from_pdf(filename)
        if raw_text:
            process_and_upload(raw_text, filename, "General Knowledge")
        
    print("\n🎉 ALL SOURCES PROCESSED!")