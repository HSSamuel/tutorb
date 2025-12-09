from fastapi import FastAPI, Form, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import re
from urllib.parse import quote 

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
    """Finds the cultural analogy. Falls back to General Wisdom if no match found."""
    supabase = tools["supabase"]
    embeddings = tools["embeddings"]
    
    # 1. Try Specific Search
    try:
        query_vector = embeddings.embed_query(subject)
        response = supabase.rpc(
            "match_cultural_knowledge",
            # Lower threshold slightly to find more loose matches
            {"query_embedding": query_vector, "match_threshold": 0.20, "match_count": 1}
        ).execute()

        if response.data and len(response.data) > 0:
            match = response.data[0]
            return f"Use this local metaphor: {match['content']} (Region: {match['region']})", match.get('image_url')
    except Exception as e:
        print(f"⚠️ Vector Search Error: {e}")

    # 2. Fallback: Fetch Random General Wisdom (If specific search failed)
    try:
        fallback = supabase.table("cultural_knowledge")\
            .select("*")\
            .contains("metadata", '{"topic": "General"}')\
            .limit(1)\
            .execute()
            
        if fallback.data:
            row = fallback.data[0]
            return f"Use this general African wisdom: {row['content']} (Region: {row['region']})", None
    except Exception as e:
        print(f"⚠️ Fallback Search Error: {e}")

    # 3. Final Resort
    return "No specific local metaphor found. Improvising based on general Nigerian culture.", None

# --- HELPER: INJECT DYNAMIC IMAGES ---
def process_text_images(text):
    """
    Finds tags like [IMAGE: A cat sitting] and replaces them with 
    real markdown image links using Pollinations AI with high-def prompts.
    """
    def replace_match(match):
        desc = match.group(1)
        encoded_desc = quote(f"educational vector diagram of {desc}, highly detailed, sharp focus, 4k resolution, svg style, white background, clean lines, minimal text")
        image_url = f"https://image.pollinations.ai/prompt/{encoded_desc}?nologo=true&width=1024&height=600"
        return f"\n\n![{desc}]({image_url})\n\n"

    return re.sub(r'\[IMAGE: (.*?)\]', replace_match, text)

# --- REUSABLE BRAIN FUNCTION (UPDATED WITH GRIOT MODE) ---
def ask_the_brain(subject: str, language: str = "english", mode: str = "standard", is_whatsapp: bool = False):
    try:
        tools = get_ai_tools()
        llm = tools["llm"]
        ChatPromptTemplate = tools["PromptTemplate"]

        # 1. Get Context & Image
        local_context, visual_url = get_context(subject, tools)
        
        # 2. IMAGE GENERATION FALLBACK
        if not visual_url:
            img_prompt = quote(f"Educational vector diagram explaining {subject}, highly detailed, sharp focus, 8k resolution, svg style, white background, clean lines, high quality, minimal text")
            visual_url = f"https://image.pollinations.ai/prompt/{img_prompt}?nologo=true&width=1024&height=600"

        # 3. DEFINE SYSTEM PROMPT BASED ON MODE
        if mode == "griot":
            # --- GRIOT MODE (Storytelling) ---
            system_instruction = """
            You are 'Baba Agba', an ancient African Griot (Storyteller).
            DO NOT explain like a textbook.
            INSTEAD, weave a short, engaging folktale involving Nigerian animals (Tortoise/Ijapa, Lion, Elephant) or village life to explain: {subject}.
            
            Structure:
            1. Start with a traditional opener like "Story, story...!" or "Under the Baobab tree..."
            2. Tell the fable where the scientific concept is the main plot device (e.g., Friction is why Tortoise stopped sliding).
            3. End with "And that is why..." linking the story back to the science definition.
            
            Tone: Wise, rhythmic, engaging. Use atmospheric descriptions.
            """
        else:
            # --- STANDARD MODE (Tutor) ---
            system_instruction = """
            You are a Nigerian science tutor. 
            Explain: {subject} clearly.
            Use this context if relevant: {context}
            Tone: Friendly, academic but accessible.
            """

        # 4. DEFINE TONE (Pidgin vs English)
        tone_instruction = ""
        if language == "pidgin":
            tone_instruction = "Speak in Nigerian Pidgin English. Use slang like 'Abeg', 'No wahala', 'Omo' to make it fun."
        else:
            tone_instruction = "Speak in clear, standard English."

        # 5. DEFINE FORMATTING
        if is_whatsapp:
             formatting_instruction = "Start with a friendly emoji. Use single * for bold. Use dashes for lists. No LaTeX. Keep under 100 words."
        else:
             formatting_instruction = """
             Use **Bold** for key terms. Use LaTeX for math ($E=mc^2$). 
             If you describe a specific physical concept that needs visualization, 
             insert an image tag like this: [IMAGE: short description].
             """

        # 6. Build Prompt
        prompt = ChatPromptTemplate.from_template(f"""
        {system_instruction}
        
        TONE INSTRUCTION: {tone_instruction}
        
        FORMATTING:
        {formatting_instruction}
        
        Student asks: {{subject}}
        Context: {{context}}
        """)

        # 7. Invoke Chain
        chain = prompt | llm
        ai_reply = chain.invoke({
            "subject": subject, 
            "context": local_context
        })
        
        # 8. Process Output
        final_answer = ai_reply.content
        if not is_whatsapp:
            final_answer = process_text_images(final_answer)
        else:
            final_answer = clean_for_whatsapp(final_answer)
        
        return final_answer, local_context, visual_url

    except Exception as e:
        print(f"❌ Error: {e}")
        return f"Error: {str(e)}", "Error", None

# --- ENDPOINTS ---

@app.get("/")
def home():
    return {"status": "AI Tutor Brain is Online (Griot Mode Available) 🧠✨📜"}

class TopicRequest(BaseModel):
    subject: str
    language: str = "english"
    mode: str = "standard" # <--- Added Mode

# 1. TEACH ENDPOINT
@app.post("/teach")
async def teach_topic(request: TopicRequest):
    # Pass request.mode to the brain
    answer, context, image = ask_the_brain(
        request.subject, 
        language=request.language, 
        mode=request.mode, 
        is_whatsapp=False
    )
    return {
        "response": answer,
        "source_data": context,
        "visual_aid": image
    }

# 2. QUIZ ENDPOINT
@app.post("/quiz")
async def generate_quiz(request: TopicRequest):
    print(f"❓ Generating Contextual Quiz for: {request.subject}")
    try:
        tools = get_ai_tools()
        llm = tools["llm"]
        ChatPromptTemplate = tools["PromptTemplate"]
        
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
            "context": local_context
        })
        
        return {"quiz": ai_reply.content}
        
    except Exception as e:
        return {"error": str(e)}

# 3. WHATSAPP ENDPOINT
@app.post("/whatsapp")
async def whatsapp_reply(Body: str = Form(...)):
    print(f"📩 WhatsApp Message: {Body}")
    # WhatsApp defaults to Standard mode
    answer, context, image = ask_the_brain(Body, is_whatsapp=True)
    
    xml_response = "<Response><Message>"
    xml_response += f"<Body>{answer}</Body>"
    if image:
        xml_response += f"<Media>{image}</Media>"
    xml_response += "</Message></Response>"
    
    return Response(content=xml_response, media_type="application/xml")