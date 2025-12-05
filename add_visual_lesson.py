import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from supabase import create_client, Client

# Setup
load_dotenv()
supabase: Client = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SERVICE_KEY"))
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def upload_visual_lesson(text, region, image_link):
    print(f"🎨 Embedding visual lesson: {text[:30]}...")
    vector = embeddings.embed_query(text)
    
    data = {
        "content": text,
        "region": region,
        "embedding": vector,
        "image_url": image_link  # <--- SAVING THE IMAGE LINK
    }
    
    supabase.table("cultural_knowledge").insert(data).execute()
    print("✅ Uploaded successfully!")

if __name__ == "__main__":
    # DEFINING A LESSON WITH AN IMAGE
    
    # 1. The Image URL (A clear diagram of Ohm's Law/Circuit)
    # Note: In a real app, you'd upload this to Supabase Storage. For now, we use a public educational URL.
    circuit_diagram = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d2/Ohm%27s_Law_with_Voltage_Source.svg/640px-Ohm%27s_Law_with_Voltage_Source.svg.png"

    # 2. The Text Explanation
    explanation = """
    In a simple electrical circuit, imagine the Battery is the 'Pump' pushing water (Current) through a narrow pipe (Resistance).
    The image provided shows a basic circuit diagram where V is Voltage, I is Current, and R is the Resistor.
    This clearly illustrates Ohm's Law: V = I * R.
    (Source: Visual Physics DB)
    """

    upload_visual_lesson(explanation, "Visual Diagram", circuit_diagram)