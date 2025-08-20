import os
import json
import re
import time
import cohere
import httpx
import pandas as pd
from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from vertexai import init
from vertexai.generative_models import GenerativeModel
from util.reportGen import generate_html_report, generate_html_reportRag

# Updated imports for LangChain 0.2+ (no deprecation warnings)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ==========================================================
# Load environment variables
# ==========================================================
load_dotenv()

# ==========================================================
# VectorDB Loader
# ==========================================================
PERSIST_DIR = r"C:\Users\VM116ZZ\PycharmProjects\POC\vectordb"
TOP_K = 3

def load_vectordb():
    embeddings = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL_NAME"))
    vectordb = Chroma(
        collection_name="rag_contexts",
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )
    return vectordb.as_retriever(search_kwargs={"k": TOP_K})

retriever = load_vectordb()

# ==========================================================
# LLM PROVIDER WRAPPER — plug & play
# ==========================================================
def call_llm(provider, model_name, prompt, retries=5, wait_seconds=15):
    print("Using Provider : ", provider)
    print("Using Model : ", model_name)

    provider = provider.lower()

    if provider == "gemini":
        PROJECT_ID = os.getenv("PROJECT_ID")
        LOCATION = os.getenv("LOCATION")
        VERTEX_API_KEY_FILENAME = os.getenv("VERTEX_APIKEY_FILE_NAME")
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        SERVICE_ACCOUNT_PATH = os.path.join(BASE_DIR, "resources", "API_Keys", "VertexAPIKey", VERTEX_API_KEY_FILENAME)

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_PATH
        init(project=PROJECT_ID, location=LOCATION)

        for attempt in range(retries):
            try:
                model = GenerativeModel(model_name)
                return model.generate_content(prompt).text
            except ResourceExhausted:
                if attempt < retries - 1:
                    print(f"[WARN] Gemini resource exhausted. Retrying in {wait_seconds} seconds... (Attempt {attempt + 1}/{retries})")
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

    elif provider == "cohere":
        original_client_init = httpx.Client.__init__

        def client_init_no_ssl(self, *args, **kwargs):
            kwargs["verify"] = False
            original_client_init(self, *args, **kwargs)

        try:
            httpx.Client.__init__ = client_init_no_ssl
            co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
            resp = co.chat(model=model_name, message=prompt)
            return resp.text
        finally:
            httpx.Client.__init__ = original_client_init

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

# ==========================================================
# Load test data
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
# RAG Evaluation
# ==========================================================
test_data_file = "resources/TestData/llm_eval_test_data.csv"
df = load_test_data(test_data_file)

embedder = SentenceTransformer(os.getenv("EMBEDDING_MODEL_NAME"))

for idx, row in df.iterrows():
    question = row["question"]
    expected_answer = row["expected_answer"]
    llm_response = row["llm_response"]

    print(f"\nProcessing row {idx+1}/{len(df)}: {question}")

    # Retrieve context dynamically from Chroma
    retrieved_docs = retriever.invoke(question)
    context = " ".join(doc.page_content for doc in retrieved_docs)

    # =========================
    # NEW: Context Precision Score
    # =========================
    embeddings = embedder.encode([question, expected_answer, context])
    sim_context_question = cosine_similarity([embeddings[2]], [embeddings[0]])[0][0]
    sim_context_answer = cosine_similarity([embeddings[2]], [embeddings[1]])[0][0]
    context_precision_score = (sim_context_question + sim_context_answer) / 2

    # Semantic Similarity Score (Expected vs LLM)
    embeddings_response = embedder.encode([expected_answer, llm_response])
    similarity_score = cosine_similarity([embeddings_response[0]], [embeddings_response[1]])[0][0]

    # Faithfulness to Context
    faithfulness_prompt = f"""
    Context: {context}
    LLM Response: {llm_response}

    On a scale of 0 to 1, rate how factually faithful the LLM's response is to the provided context.
    Return ONLY a number between 0 and 1.
    """
    # faithfulness_eval = call_llm("gemini", os.getenv("GEMINI_MODEL_NAME"), faithfulness_prompt)
    faithfulness_eval = call_llm("cohere", os.getenv("COHERE_MODEL_NAME"), faithfulness_prompt)

    try:
        faithfulness_score = float(faithfulness_eval.strip())
    except ValueError:
        faithfulness_score = 0.0

    # Reasoning Check
    reasoning_prompt = f"""
    Question: {question}
    Context: {context}
    Expected Answer: {expected_answer}
    LLM Response: {llm_response}

    Considering the context, rate the LLM response on a scale of 0 to 1 for correctness, factuality, and relevance.
    Return ONLY a number between 0 and 1.
    """
    # reasoning_eval = call_llm("gemini", os.getenv("GEMINI_MODEL_NAME"), reasoning_prompt)
    reasoning_eval = call_llm("cohere", os.getenv("COHERE_MODEL_NAME"), reasoning_prompt)

    try:
        reasoning_score = float(reasoning_eval.strip())
    except ValueError:
        reasoning_score = 0.0

    # Detailed Metrics (RAG-focused)
    metrics_prompt = f"""
    You are an evaluator. Analyze the given LLM response using RAG evaluation principles.

    Question: {question}
    Context: {context}
    Expected Answer: {expected_answer}
    LLM Response: {llm_response}

    Evaluate and return ONLY valid JSON with:
    - correctness: integer 0-10
    - completeness: integer 0-10
    - clarity: integer 0-10
    - hallucination: integer 0-10
    - faithfulness_to_context: integer 0-10
    - relevance: integer 0-10
    - conciseness: integer 0-10
    - comments: short constructive feedback
    """
    # metrics_eval = call_llm("gemini", os.getenv("GEMINI_MODEL_NAME"), metrics_prompt)
    metrics_eval = call_llm("cohere", os.getenv("COHERE_MODEL_NAME"), metrics_prompt)

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

    # Store in DataFrame
    df.at[idx, "context_precision_score"] = context_precision_score
    df.at[idx, "similarity_score"] = similarity_score
    df.at[idx, "faithfulness_score"] = faithfulness_score
    df.at[idx, "reasoning_score"] = reasoning_score
    df.at[idx, "metrics_json"] = json.dumps(metrics_json)

    # Print for debugging
    print(f"Context Precision Score: {context_precision_score:.2f}")
    print(f"Similarity Score: {similarity_score:.2f}")
    print(f"Faithfulness Score: {faithfulness_score:.2f}")
    print(f"Reasoning Score: {reasoning_score:.2f}")
    print(f"Metrics: {metrics_json}")

# ==========================================================
# Generate Report
# ==========================================================
generate_html_reportRag(df, "reports/RAG_Evaluator_report.html")