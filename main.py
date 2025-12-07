from fastapi import FastAPI, Form, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import re # <--- NEW IMPORT for text cleaning

# 1. Load Env Vars
load_dotenv()

# 2. Setup App
app = FastAPI()

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
        
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
        cohere_key = os.environ.get("COHERE_API_KEY")
        
        supabase = create_client(supabase_url, supabase_key)
        
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
    return ai_tools

# --- HELPER: WHATSAPP CLEANER ---
def clean_for_whatsapp(text):
    """
    Converts AI Markdown to WhatsApp-friendly format.
    """
    # 1. Convert Double Bold (**text**) to WhatsApp Bold (*text*)
    text = text.replace("**", "*")
    
    # 2. Convert Headers (### Header) to Uppercase Bold (*HEADER*)
    # This finds hashtags, removes them, and wraps the text in stars
    text = re.sub(r'#+\s*(.*)', r'*\1*', text)
    
    # 3. Remove LaTeX Math ($...$) symbols, keep the equation
    text = text.replace("$", "")
    
    return text.strip()

# --- REUSABLE BRAIN FUNCTION ---
def ask_the_brain(subject: str, is_whatsapp: bool = False):
    try:
        tools = get_ai_tools()
        supabase = tools["supabase"]
        embeddings = tools["embeddings"]
        llm = tools["llm"]
        ChatPromptTemplate = tools["PromptTemplate"]

        # 1. Search
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

        # 2. Generate Prompt (UPDATED FOR BETTER FORMATTING)
        if is_whatsapp:
             formatting_instruction = """
             1. Start with a friendly emoji (e.g. 📚 or 💡).
             2. Use SINGLE asterisks for bold keys: *like this* (NOT double **).
             3. Use dashes (-) for list items.
             4. NO LaTeX math. Write formulas plainly: E = mc^2.
             5. Structure: 
                *Metaphor*
                (Explanation)
                *Scientific Fact*
                (Explanation)
             6. Keep it under 100 words.
             """
        else:
             formatting_instruction = "Use **Bold** for key terms. Use LaTeX for math ($E=mc^2$). Keep it under 150 words."

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
        
        # Clean text if it's for WhatsApp
        final_answer = clean_for_whatsapp(ai_reply.content) if is_whatsapp else ai_reply.content
        
        return final_answer, local_context, visual_url

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
    # Web uses is_whatsapp=False (Keeps Rich Markdown)
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
    
    # WhatsApp uses is_whatsapp=True (Clean Text)
    answer, context, image = ask_the_brain(Body, is_whatsapp=True)
    
    # Twilio XML Response
    xml_response = "<Response><Message>"
    xml_response += f"<Body>{answer}</Body>"
    if image:
        xml_response += f"<Media>{image}</Media>"
    xml_response += "</Message></Response>"
    
    return Response(content=xml_response, media_type="application/xml")