import os
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import json
import re

# ==========================================================
# Load environment variables
# ==========================================================
load_dotenv()

# ==========================================================
#  LLM PROVIDER WRAPPER — plug & play
# ==========================================================
def call_llm(provider, model_name, prompt):
    provider = provider.lower()

    if provider == "gemini":
        from vertexai import init
        from vertexai.generative_models import GenerativeModel

        PROJECT_ID = os.getenv("PROJECT_ID")
        LOCATION = os.getenv("LOCATION")
        VERTEX_API_KEY_FILENAME = os.getenv("VERTEX_APIKEY_FILE_NAME")
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        SERVICE_ACCOUNT_PATH = os.path.join(BASE_DIR, "resources", "API_Keys", "VertexAPIKey", VERTEX_API_KEY_FILENAME)

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_PATH
        init(project=PROJECT_ID, location=LOCATION)

        model = GenerativeModel(model_name)
        return model.generate_content(prompt).text

    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an evaluator."},
                {"role": "user", "content": prompt}
            ]
        )
        return resp.choices[0].message["content"]

    elif provider == "huggingface":
        from transformers import pipeline
        generator = pipeline("text-generation", model=model_name)
        return generator(prompt, max_length=500)[0]["generated_text"]

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

# -------------------------
# Sample Data
# -------------------------
question = "What is the capital of France?"
expected_answer = "The capital of France is Paris."
llm_response = "Paris"

# -------------------------
# Semantic Similarity Score
# -------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode([expected_answer, llm_response])
similarity_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

# -------------------------
# Reasoning Check
# -------------------------
reasoning_prompt = f"""
Question: {question}
Expected Answer: {expected_answer}
LLM Response: {llm_response}

Rate the LLM response on a scale of 0 to 1 for correctness, factuality, and relevance.
Return ONLY a number between 0 and 1.
"""

gemini_eval = call_llm("gemini", "gemini-2.0-flash-001", reasoning_prompt)  # ⬅ Change provider/model here
try:
    reasoning_score = float(gemini_eval.strip())
except ValueError:
    reasoning_score = 0.0

# -------------------------
# Detailed Metric Evaluation (Expanded Metrics)
# -------------------------
metrics_prompt = f"""
You are an evaluator. Analyze the given LLM response.

Question: {question}
Expected Answer: {expected_answer}
LLM Response: {llm_response}

Evaluate the response and return ONLY valid JSON with:
- correctness: integer 0-10
- completeness: integer 0-10
- clarity: integer 0-10
- hallucination: integer 0-10
- faithfulness: integer 0-10
- relevance: integer 0-10
- conciseness: integer 0-10
- comments: short constructive feedback

Example:
{{
    "correctness": 9,
    "completeness": 8,
    "clarity": 10,
    "hallucination": 0,
    "faithfulness": 9,
    "relevance": 10,
    "conciseness": 9,
    "comments": "Accurate and well phrased. Minimal unnecessary detail."
}}
"""

# Change provider/model here
metrics_eval = call_llm("gemini", "gemini-2.0-flash-001", metrics_prompt)
raw_text = metrics_eval.strip()

# Step 1: Remove Markdown code fences if present
if raw_text.startswith("```"):
    raw_text = re.sub(r"^```(?:json)?", "", raw_text, flags=re.IGNORECASE).strip()
    raw_text = re.sub(r"```$", "", raw_text).strip()

# Step 2: Extract JSON inside curly braces
match = re.search(r"\{.*\}", raw_text, re.DOTALL)
if match:
    raw_text = match.group(0)

# Step 3: Parse JSON safely
try:
    metrics = json.loads(raw_text)
except json.JSONDecodeError:
    print("LLM returned non-JSON output, falling back to defaults.")
    metrics = {
        "correctness": 0, "completeness": 0, "clarity": 0,
        "hallucination": 0, "faithfulness": 0, "relevance": 0, "conciseness": 0,
        "comments": f"Parsing error. Raw output: {metrics_eval[:100]}..."
    }

# -------------------------
# Final Evaluation
# -------------------------
final_score = (similarity_score + reasoning_score) / 2

print(f"Semantic Similarity Score: {similarity_score:.2f}")
print(f"Reasoning Score: {reasoning_score:.2f}")
print(f"Final Evaluation Score: {final_score:.2f}")
print("\n--- Detailed Metrics ---")
for key, value in metrics.items():
    if isinstance(value, int):
        print(f"{key.capitalize()}: {value}/10")
    else:
        print(f"{key.capitalize()}: {value}")