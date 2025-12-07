from fastapi import FastAPI, Form, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# 1. Load Env Vars
load_dotenv()

# 2. Setup App
app = FastAPI()

# "Public Mode" CORS - Allows everyone (Bulletproof for MVP)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. GLOBAL VARIABLES
ai_tools = None

# --- HELPER: CLOUD LOADER ---
def get_ai_tools():
    global ai_tools
    
    if ai_tools is None:
        print("⏳ Connecting to Cloud AI...")
        from langchain_cohere import ChatCohere, CohereEmbeddings
        from langchain_core.prompts import ChatPromptTemplate
        from supabase import create_client
        
        # Load Keys
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
        cohere_key = os.environ.get("COHERE_API_KEY")
        
        # Initialize
        supabase = create_client(supabase_url, supabase_key)
        
        # LIGHTWEIGHT: Use Cloud Embeddings (No RAM Usage)
        print("⏳ Connecting to Cohere Embeddings API...")
        embeddings = CohereEmbeddings(
            model="embed-english-light-v3.0", 
            cohere_api_key=cohere_key
        )
        
        print("⏳ Connecting to Cohere Chat API...")
        llm = ChatCohere(model="command-r-08-2024", cohere_api_key=cohere_key)
        
        ai_tools = {
            "supabase": supabase,
            "embeddings": embeddings,
            "llm": llm,
            "PromptTemplate": ChatPromptTemplate
        }
        print("✅ AI Tools Ready!")
        
    return ai_tools

# --- REUSABLE BRAIN FUNCTION ---
def ask_the_brain(subject: str, is_whatsapp: bool = False):
    """
    Core logic: Search Supabase -> Ask Cohere -> Return Answer
    """
    try:
        tools = get_ai_tools()
        supabase = tools["supabase"]
        embeddings = tools["embeddings"]
        llm = tools["llm"]
        ChatPromptTemplate = tools["PromptTemplate"]

        # 1. Search (Retrieval)
        query_vector = embeddings.embed_query(subject)
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

        # 2. Generate Prompt
        # We adjust instructions slightly for WhatsApp (shorter) vs Web (richer)
        formatting_instruction = (
            "Keep it short (under 100 words). Use *bold* for emphasis." 
            if is_whatsapp 
            else "Use **Bold** for key terms. Use LaTeX for math ($E=mc^2$). Keep it under 150 words."
        )

        prompt = ChatPromptTemplate.from_template("""
        You are a Nigerian science tutor. Explain clearly.
        Student asks: {subject}
        Context: {context}
        
        INSTRUCTIONS:
        {formatting}
        If an image is provided in context, refer to "the diagram".
        """)

        chain = prompt | llm
        ai_reply = chain.invoke({
            "subject": subject, 
            "context": local_context,
            "formatting": formatting_instruction
        })
        
        return ai_reply.content, local_context, visual_url

    except Exception as e:
        print(f"❌ Error: {e}")
        return f"Error: {str(e)}", "Error", None

# --- ENDPOINTS ---

@app.get("/")
def home():
    return {"status": "AI Tutor Brain is Online (Cloud + WhatsApp Ready) ☁️📱"}

# 1. WEB ENDPOINT (JSON)
class TopicRequest(BaseModel):
    subject: str

@app.post("/teach")
async def teach_topic(request: TopicRequest):
    print(f"🔍 Web Request: {request.subject}")
    answer, context, image = ask_the_brain(request.subject, is_whatsapp=False)
    
    return {
        "response": answer,
        "source_data": context,
        "visual_aid": image,
        "model_used": "Cohere Command-R"
    }

# 2. WHATSAPP ENDPOINT (XML)
@app.post("/whatsapp")
async def whatsapp_reply(Body: str = Form(...)):
    print(f"📩 WhatsApp Message: {Body}")
    
    answer, context, image = ask_the_brain(Body, is_whatsapp=True)
    
    # Twilio XML Response
    xml_response = "<Response><Message>"
    xml_response += f"<Body>{answer}</Body>"
    if image:
        xml_response += f"<Media>{image}</Media>"
    xml_response += "</Message></Response>"
    
    return Response(content=xml_response, media_type="application/xml")