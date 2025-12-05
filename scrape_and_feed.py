import os
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader  # <--- NEW: For reading PDFs
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import create_client, Client

# 1. Setup & Config
load_dotenv()
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

print("⏳ Loading AI Model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

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
    """Chunks text and uploads to Supabase"""
    if not text or len(text) < 100:
        print("   ⚠️ Text too short or empty. Skipping.")
        return

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # Larger chunks for textbooks
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = text_splitter.split_text(text)
    
    print(f"   🔪 Split into {len(chunks)} chunks. Uploading...")

    # Upload Loop
    for i, chunk in enumerate(chunks):
        vector = embeddings.embed_query(chunk)
        
        # We append the source to the content so the AI knows where it learned this
        final_content = f"{chunk}\n(Source: {source_name})"
        
        supabase.table("cultural_knowledge").insert({
            "content": final_content,
            "region": region_tag,
            "embedding": vector
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