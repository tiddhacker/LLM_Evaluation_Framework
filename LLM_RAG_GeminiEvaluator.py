import os

from sklearn.metrics.pairwise import cosine_similarity

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] = ""
from dotenv import load_dotenv
import time
import sys
import grpc
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.llms.base import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    answer_similarity,
    faithfulness,
    answer_correctness
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma                                                     # <---- ADDED


# Load environment variables from .env file
load_dotenv()
VERTEX_API_KEY_FILENAME = os.getenv("VERTEX_APIKEY_FILE_NAME")

# Config for loading LLM key
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SERVICE_ACCOUNT_PATH = os.path.join(BASE_DIR, "resources", "API_Keys", "VertexAPIKey", VERTEX_API_KEY_FILENAME)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_PATH

#this is to ignore some unwanted logs
def silent_excepthook(exc_type, exc_value, exc_traceback):
    # Ignore noisy gRPC shutdown warnings
    if "grpc_wait_for_shutdown" in str(exc_value):
        return
    print(f"Unhandled exception: {exc_value}")
sys.excepthook = silent_excepthook

PERSIST_DIR = r"C:\Users\VM116ZZ\PycharmProjects\POC\vectordb"
TOP_K = 3
# LOAD CHROMA AS RETRIEVER
def load_vectordb():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(
        collection_name="rag_contexts",
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )
    return vectordb.as_retriever(search_kwargs={"k": TOP_K})


def get_llm_and_metrics():
    import vertexai
    import google.auth
    from langchain_google_vertexai import VertexAI, VertexAIEmbeddings

    if not PROJECT_ID or not LOCATION:
        raise ValueError("GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION must be set in .env")

    vertexai.init(project=PROJECT_ID, location=LOCATION)
    creds, _ = google.auth.default()
    print("Resolved credentials path:", creds._service_account_email if hasattr(creds, '_service_account_email') else creds)

    llm = VertexAI(model_name=os.getenv("GEMINI_MODEL_NAME"), credentials=creds)
    wrapper = LangchainLLMWrapper(llm)

    class RAGASVertexAIEmbeddings(VertexAIEmbeddings):
        async def embed_text(self, text: str) -> list[float]:
            return self.embed([text], 1, "SEMANTIC_SIMILARITY")[0]

        def set_run_config(self, *args, **kwargs):
            pass

    embeddings = RAGASVertexAIEmbeddings(model_name="text-embedding-005", credentials=creds)

    metrics = [answer_relevancy, context_recall, context_precision, answer_similarity, faithfulness, answer_correctness]
    for m in metrics:
        m.llm = wrapper
        if hasattr(m, "embeddings"):
            m.embeddings = embeddings

    return wrapper, embeddings, metrics


def evaluate_with_retries(index: int, dataset: Dataset, wrapper, embeddings, metrics, max_retries=3, delay=5):
    retries = 0
    while retries < max_retries:
        try:
            selected_data = dataset.select([index])
            result = evaluate(
                selected_data,
                metrics=metrics,
                llm=wrapper,
                embeddings=embeddings
            ).to_pandas()
            return result
        except grpc.RpcError as e:
            print(f"gRPC Error: {e}. Retrying {retries + 1}/{max_retries}...")
            time.sleep(delay * (2 ** retries))
            retries += 1
        except Exception as e:
            print(f"Unexpected Error: {e}")
            sys.exit(1)
    print("Evaluation failed after retries.")
    return None


async def evaluate_single_question(question, answer, reference):
    # 1) Load vector DB
    retriever = load_vectordb()

    # 2) Retrieve top-k relevant chunks
    retrieved_docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in retrieved_docs]
    source_docs = [doc.metadata.get("source", "") for doc in retrieved_docs]

    print(f"\nTop-{TOP_K} retrieved context chunks for question: '{question}'")
    for idx, doc in enumerate(retrieved_docs, start=1):
        print(f"  [{idx}] Source: {doc.metadata.get('source', '')}")
        print(f"      Chunk: {doc.page_content[:150]}...\n")  # printing only first 150 chars for readability

    # 3) Convert to RAGAS data-sample format
    data_samples = {
        'question': [question],
        'answer': [answer],
        'contexts': [contexts],
        'reference': [reference],
        'ground_truth': [reference],
        'sources': [source_docs]

    }

    print("Initializing LLM and metrics...")
    wrapper, embeddings, metrics = get_llm_and_metrics()

    results_set = []
    dataset = Dataset.from_dict(data_samples)
    for i in range(len(dataset)):
        print(f"--- Evaluating sample #{i + 1}/{len(dataset)} ---")
        result = evaluate_with_retries(i, dataset, wrapper, embeddings, metrics)
        if result is not None:
          results_set.append(result)

    if results_set:
        results_df = pd.concat(results_set)
        if "retrieved_contexts" in results_df.columns:
            results_df = results_df.drop(columns=["retrieved_contexts"])
    else:
        print("No evaluation completed.")

    return results_set

#===================================================================
#======================CHECK CONSISTENCY============================
#===================================================================

async def check_consistency(answers: list[str], n_runs=3):
    """
    Check consistency between multiple answers for the same question.
    - answers: list of candidate answers (strings)
    - n_runs: how many answers to check
    """
    print("Initializing embeddings...")
    _, embeddings, _ = get_llm_and_metrics()

    if len(answers) < n_runs:
        raise ValueError(f"Need at least {n_runs} answers, but got {len(answers)}")

    # Take only the first n_runs answers
    selected_answers = answers[:n_runs]

    # Embed all answers
    embedded = embeddings.embed_documents(selected_answers)

    # Compute pairwise cosine similarities
    sims = []
    for i in range(len(embedded)):
        for j in range(i + 1, len(embedded)):
            sims.append(cosine_similarity([embedded[i]], [embedded[j]])[0][0])

    return sum(sims) / len(sims) if sims else 1.0