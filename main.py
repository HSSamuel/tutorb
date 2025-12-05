from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
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

@app.get("/")
def home():
    return {"status": "AI Tutor Brain is Online 🧠", "version": "Multi-Modal"}

# Allow Frontend to talk to Backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Setup Database & AI
supabase: Client = create_client(supabase_url, supabase_key)

print("⏳ Loading Local Embeddings Model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("⏳ Connecting to Cohere AI...")
# Using the stable model version we found earlier
llm = ChatCohere(model="command-r-08-2024", cohere_api_key=cohere_key)

class TopicRequest(BaseModel):
    subject: str

@app.post("/teach")
async def teach_topic(request: TopicRequest):
    print(f"🔍 Student asked about: {request.subject}")

    try:
        # A. SEARCH
        query_vector = embeddings.embed_query(request.subject)
        
        # Note: We reset threshold to 0.0 to ensure we find our new visual entry
        response = supabase.rpc(
            "match_cultural_knowledge",
            {"query_embedding": query_vector, "match_threshold": 0.0, "match_count": 1}
        ).execute()

        local_context = "No specific local metaphor found."
        visual_url = None  # <--- DEFAULT: No image

        if response.data:
            match = response.data[0]
            local_context = f"Use this local metaphor: {match['content']} (Region: {match['region']})"
            
            # 👇 CAPTURE THE IMAGE URL
            visual_url = match.get('image_url') 
            
            print(f"✅ Found context: {match['content'][:30]}...")

        # B. GENERATE (Prompt remains mostly same)
        prompt = ChatPromptTemplate.from_template("""
        You are a Nigerian science tutor. 
        The student wants to know about: {subject}
        
        STRICTLY use this local context to explain it:
        {context}
        
        FORMATTING:
        1. Use **Bold** for key terms.
        2. Use LaTeX for math ($E=mc^2$).
        3. If an image is provided in the context, refer to "the diagram shown".
        """)

        chain = prompt | llm
        
        ai_reply = chain.invoke({
            "subject": request.subject,
            "context": local_context
        })
        
        return {
            "response": ai_reply.content,
            "source_data": local_context,
            "visual_aid": visual_url,  # <--- SEND IMAGE TO FRONTEND
            "model_used": "Cohere Command-R"
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        return {"error": str(e)}