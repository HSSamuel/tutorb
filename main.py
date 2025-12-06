from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# 1. Load Env Vars
load_dotenv()

# 2. Setup App (Lightweight Startup)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # Allow ALL origins (Netlify, Localhost, www, etc.)
    allow_credentials=False, # <--- Turn this OFF to allow "*"
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. GLOBAL VARIABLES
# We do NOT initialize them here. We wait until the first request.
ai_tools = None

# --- HELPER: ULTRA-LAZY LOADER ---
def get_ai_tools():
    """
    Imports and loads AI tools ONLY when needed.
    This makes the server start instantly (<1 second).
    """
    global ai_tools
    
    if ai_tools is None:
        print("⏳ Importing Heavy Libraries (First Run Only)...")
        # 👇 HEAVY IMPORTS MOVED INSIDE HERE
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_cohere import ChatCohere
        from langchain_core.prompts import ChatPromptTemplate
        from supabase import create_client
        
        # Load Keys
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
        cohere_key = os.environ.get("COHERE_API_KEY")
        
        # Initialize
        print("⏳ Connecting to Brain...")
        supabase = create_client(supabase_url, supabase_key)
        
        print("⏳ Loading Embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        print("⏳ Loading Cohere...")
        llm = ChatCohere(model="command-r-08-2024", cohere_api_key=cohere_key)
        
        # Save to global variable
        ai_tools = {
            "supabase": supabase,
            "embeddings": embeddings,
            "llm": llm,
            "PromptTemplate": ChatPromptTemplate
        }
        print("✅ AI Tools Ready!")
        
    return ai_tools

# --- ENDPOINTS ---

@app.get("/")
def home():
    # This endpoint is super fast, proving to Render that we are "Live"
    return {"status": "AI Tutor Brain is Online 🧠", "mode": "Ultra-Lazy Loading"}

class TopicRequest(BaseModel):
    subject: str

@app.post("/teach")
async def teach_topic(request: TopicRequest):
    print(f"🔍 Student asked about: {request.subject}")

    try:
        # 1. LOAD TOOLS NOW
        tools = get_ai_tools()
        
        # Unpack tools
        supabase = tools["supabase"]
        embeddings = tools["embeddings"]
        llm = tools["llm"]
        ChatPromptTemplate = tools["PromptTemplate"]

        # A. SEARCH (Retrieval)
        query_vector = embeddings.embed_query(request.subject)
        
        response = supabase.rpc(
            "match_cultural_knowledge",
            {"query_embedding": query_vector, "match_threshold": 0.0, "match_count": 1}
        ).execute()

        local_context = "No specific local metaphor found."
        visual_url = None

        if response.data:
            match = response.data[0]
            local_context = f"Use this local metaphor: {match['content']} (Region: {match['region']})"
            visual_url = match.get('image_url')
            print(f"✅ Found context: {match['content'][:30]}...")

        # B. GENERATE
        prompt = ChatPromptTemplate.from_template("""
        You are a Nigerian science tutor. You explain things clearly and structured.
        
        The student wants to know about: {subject}
        
        STRICTLY use this local context to explain it:
        {context}
        
        INSTRUCTIONS FOR FORMATTING:
        1. Use **Bold** for key terms.
        2. Use Bullet points for steps or lists.
        3. Use LaTeX for math ($E=mc^2$).
        4. If an image is provided in the context, refer to "the diagram shown".
        5. Keep the explanation engaging but concise (under 150 words).
        
        If no context is provided, use a generic Nigerian example.
        """)

        chain = prompt | llm
        
        ai_reply = chain.invoke({
            "subject": request.subject,
            "context": local_context
        })
        
        return {
            "response": ai_reply.content,
            "source_data": local_context,
            "visual_aid": visual_url,
            "model_used": "Cohere Command-R"
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        return {"error": str(e)}