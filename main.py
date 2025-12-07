from fastapi import FastAPI, Form, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from urllib.parse import quote  # <--- NEW: To format URLs safely
from dotenv import load_dotenv
import os
import re

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
        print("✅ AI Tools Ready!")
        
    return ai_tools

# --- HELPER: WHATSAPP CLEANER ---
def clean_for_whatsapp(text):
    text = text.replace("**", "*")
    text = re.sub(r'#+\s*(.*)', r'*\1*', text)
    text = text.replace("$", "")
    return text.strip()

# --- HELPER: RETRIEVE CONTEXT ---
def get_context(subject: str, tools):
    """Finds the cultural analogy for a given subject"""
    supabase = tools["supabase"]
    embeddings = tools["embeddings"]
    
    # Embed the query
    query_vector = embeddings.embed_query(subject)
    
    # Search Database
    response = supabase.rpc(
        "match_cultural_knowledge",
        {"query_embedding": query_vector, "match_threshold": 0.0, "match_count": 1}
    ).execute()

    if response.data:
        match = response.data[0]
        # Return the rich cultural context
        return f"Use this local metaphor: {match['content']} (Region: {match['region']})", match.get('image_url')
    
    return "No specific local metaphor found.", None

# --- REUSABLE BRAIN FUNCTION ---
def ask_the_brain(subject: str, is_whatsapp: bool = False):
    try:
        tools = get_ai_tools()
        llm = tools["llm"]
        ChatPromptTemplate = tools["PromptTemplate"]

        # 1. Get Context
        local_context, visual_url = get_context(subject, tools)

        # 2. IMAGE GENERATION FALLBACK (The Magic ✨)
        # If Supabase didn't have an image, we generate one on the fly!
        if not visual_url:
            # Create a safe, encoded prompt for the image generator
            # We ask for a "textbook style" vector illustration
            img_prompt = quote(f"Educational diagram explaining {subject}, clear scientific vector illustration, white background, high quality, minimal text")
            visual_url = f"https://image.pollinations.ai/prompt/{img_prompt}?nologo=true&width=1024&height=600"

        # 3. Generate Prompt
        if is_whatsapp:
             formatting_instruction = "Start with a friendly emoji. Use single * for bold. Use dashes for lists. No LaTeX. Keep under 100 words."
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
    return {"status": "AI Tutor Brain is Online (Contextual Quiz Mode) 🧠✨"}

class TopicRequest(BaseModel):
    subject: str

# 1. TEACH ENDPOINT
@app.post("/teach")
async def teach_topic(request: TopicRequest):
    answer, context, image = ask_the_brain(request.subject, is_whatsapp=False)
    return {
        "response": answer,
        "source_data": context,
        "visual_aid": image
    }

# 2. QUIZ ENDPOINT (UPDATED FOR PRECISION)
@app.post("/quiz")
async def generate_quiz(request: TopicRequest):
    print(f"❓ Generating Contextual Quiz for: {request.subject}")
    try:
        tools = get_ai_tools()
        llm = tools["llm"]
        ChatPromptTemplate = tools["PromptTemplate"]
        
        # 1. Retrieve the SAME context used for teaching
        # This ensures the quiz asks about the Metaphor/Analogy too!
        local_context, _ = get_context(request.subject, tools)
        
        prompt = ChatPromptTemplate.from_template("""
        Generate 5 Multiple Choice Questions (MCQs) to test a student on: {subject}.
        
        CRITICAL INSTRUCTION:
        Base the questions on this specific context/analogy if available:
        "{context}"
        
        If the context mentions a Nigerian metaphor (e.g., Cooking, Traffic), 
        at least 2 questions MUST reference that metaphor to test conceptual understanding.
        
        Strictly follow this JSON-like format for easy parsing (No intro text, No Markdown):
        
        Q1: [Question text]
        A) [Option A]
        B) [Option B]
        C) [Option C]
        Answer: [Correct Letter]
        
        Q2: [Question text]
        A) [Option A]
        B) [Option B]
        C) [Option C]
        Answer: [Correct Letter]
        
        Q3: [Question text]
        A) [Option A]
        B) [Option B]
        C) [Option C]
        Answer: [Correct Letter]

        Q4: [Question text]
        A) [Option A]
        B) [Option B]
        C) [Option C]
        Answer: [Correct Letter]

        Q5: [Question text]
        A) [Option A]
        B) [Option B]
        C) [Option C]
        Answer: [Correct Letter]
        """)
        
        chain = prompt | llm
        ai_reply = chain.invoke({
            "subject": request.subject,
            "context": local_context # <--- FEEDING THE CONTEXT HERE
        })
        
        return {"quiz": ai_reply.content}
        
    except Exception as e:
        return {"error": str(e)}

# 3. WHATSAPP ENDPOINT
@app.post("/whatsapp")
async def whatsapp_reply(Body: str = Form(...)):
    print(f"📩 WhatsApp Message: {Body}")
    answer, context, image = ask_the_brain(Body, is_whatsapp=True)
    
    xml_response = "<Response><Message>"
    xml_response += f"<Body>{answer}</Body>"
    if image:
        xml_response += f"<Media>{image}</Media>"
    xml_response += "</Message></Response>"
    
    return Response(content=xml_response, media_type="application/xml")