from dotenv import load_dotenv
import os

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
print(f"✓ Groq key loaded: {groq_key[:10]}..." if groq_key else "✗ No key found")

