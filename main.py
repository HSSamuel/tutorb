from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings # <--- Ensure this is the new import
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from supabase import create_client, Client
from dotenv import load_dotenv
import os

# 1. Load Env Vars
load_dotenv()
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
cohere_key = os.environ.get("COHERE_API_KEY")

# 2. Setup App
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. GLOBAL VARIABLES (Initialized as None)
supabase: Client = create_client(supabase_url, supabase_key)
embeddings = None
llm = None

# --- HELPER: LAZY LOADER ---
def get_ai_tools():
    """
    Loads the AI models only if they aren't loaded yet.
    This prevents the server from crashing during startup on Render.
    """
    global embeddings, llm
    
    if embeddings is None:
        print("⏳ Lazy Loading Embeddings Model...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        print("✅ Embeddings Loaded!")
        
    if llm is None:
        print("⏳ Lazy Loading Cohere...")
        llm = ChatCohere(model="command-r-08-2024", cohere_api_key=cohere_key)
        print("✅ Cohere Loaded!")
        
    return embeddings, llm

# --- ENDPOINTS ---

@app.get("/")
def home():
    return {"status": "AI Tutor Brain is Online 🧠", "mode": "Lazy Loading"}

class TopicRequest(BaseModel):
    subject: str

@app.post("/teach")
async def teach_topic(request: TopicRequest):
    print(f"🔍 Student asked about: {request.subject}")

    try:
        # 1. LOAD AI TOOLS NOW (On the first request)
        ai_embeddings, ai_llm = get_ai_tools()

        # A. SEARCH (Retrieval)
        query_vector = ai_embeddings.embed_query(request.subject)
        
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
        3. If there are math formulas, write them in LaTeX format enclosed in single dollar signs (e.g. $E = mc^2$).
        4. If an image is provided in the context, refer to "the diagram shown".
        5. Keep the explanation engaging but concise (under 150 words).
        
        If no context is provided, use a generic Nigerian example.
        """)

        chain = prompt | ai_llm
        
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