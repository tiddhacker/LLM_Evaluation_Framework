import os
import json
import re
import time

import pandas as pd
from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from vertexai import init
from vertexai.generative_models import GenerativeModel

from util.reportGen import generate_html_report

# ==========================================================
# Load environment variables
# ==========================================================
load_dotenv()

# ==========================================================
#  LLM PROVIDER WRAPPER — plug & play
# ==========================================================
def call_llm(provider, model_name, prompt, retries=5, wait_seconds=15):
    provider = provider.lower()

    if provider == "gemini":
        PROJECT_ID = os.getenv("PROJECT_ID")
        LOCATION = os.getenv("LOCATION")
        VERTEX_API_KEY_FILENAME = os.getenv("VERTEX_APIKEY_FILE_NAME")
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        SERVICE_ACCOUNT_PATH = os.path.join(BASE_DIR, "resources", "API_Keys", "VertexAPIKey", VERTEX_API_KEY_FILENAME)

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_PATH
        init(project=PROJECT_ID, location=LOCATION)

        # Retry loop
        for attempt in range(retries):
            try:
                model = GenerativeModel(model_name)
                return model.generate_content(prompt).text
            except ResourceExhausted:
                if attempt < retries - 1:
                    print(
                        f"[WARN] Gemini resource exhausted. Retrying in {wait_seconds} seconds... (Attempt {attempt + 1}/{retries})")
                    time.sleep(wait_seconds)
                else:
                    print("[ERROR] Gemini max retries reached. Raising exception.")
                    raise
            except Exception as e:
                print(f"[ERROR] Gemini unexpected error: {e}")
                raise

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

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

# ==========================================================
# Load test data (CSV or JSON)
# ==========================================================
def load_test_data(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(file_path)
    elif ext == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            return pd.DataFrame(json.load(f))
    else:
        raise ValueError("Unsupported file format. Please use CSV or JSON.")

# ==========================================================
# TEST METHOD: Main evaluation loop (row-by-row)
# ==========================================================
test_data_file = "resources/TestData/llm_eval_test_data.csv"
df = load_test_data(test_data_file)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

for idx, row in df.iterrows():
    question = row["question"]
    expected_answer = row["expected_answer"]
    llm_response = row["llm_response"]

    print(f"\nProcessing row {idx+1}/{len(df)}: {question}")

    # Semantic Similarity Score
    embeddings = embedder.encode([expected_answer, llm_response])
    similarity_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    # Reasoning Check
    reasoning_prompt = f"""
    Question: {question}
    Expected Answer: {expected_answer}
    LLM Response: {llm_response}

    Rate the LLM response on a scale of 0 to 1 for correctness, factuality, and relevance.
    Return ONLY a number between 0 and 1.
    """
    gemini_eval = call_llm("gemini", "gemini-2.0-flash-lite", reasoning_prompt)
    try:
        reasoning_score = float(gemini_eval.strip())
    except ValueError:
        reasoning_score = 0.0

    # Detailed Metrics
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
    """
    metrics_eval = call_llm("gemini", os.getenv("GEMINI_MODEL_NAME"), metrics_prompt)
    raw_text = metrics_eval.strip()

    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```(?:json)?", "", raw_text, flags=re.IGNORECASE).strip()
        raw_text = re.sub(r"```$", "", raw_text).strip()

    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        try:
            metrics_json = json.loads(match.group())
        except json.JSONDecodeError:
            metrics_json = {}
    else:
        metrics_json = {}

    print(f"Similarity Score: {similarity_score:.2f}")
    print(f"Reasoning Score: {reasoning_score:.2f}")
    print(f"Metrics: {metrics_json}")

    df.at[idx, "similarity_score"] = similarity_score
    df.at[idx, "reasoning_score"] = reasoning_score
    df.at[idx, "metrics_json"] = json.dumps(metrics_json)

# ==========================================================
# Generate Report
# ==========================================================
generate_html_report(df, "reports/multiModel_Evaluator_report.html")
