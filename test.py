import os
from dotenv import load_dotenv
import google.generativeai as genai
print("Google Generative AI module is working!")

load_dotenv()

# Retrieve the Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Check if the key was loaded
if not GEMINI_API_KEY:
    raise ValueError("Gemini API key not found. Please check your .env file.")

print(f"Gemini API Key: {GEMINI_API_KEY[:5]}********")