import os
import time
import sys
import grpc
import pandas as pd
import typing as t
from dotenv import load_dotenv
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

# Load .env variables
load_dotenv()
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")

# Suppress unhandled exceptions
def silent_excepthook(exc_type, exc_value, exc_traceback):
    print(f"Unhandled exception: {exc_value}")
sys.excepthook = silent_excepthook

# Lazy load VertexAI and setup RAGAS
def get_llm_and_metrics():
    import vertexai
    import google.auth
    from langchain_google_vertexai import VertexAI, VertexAIEmbeddings

    if not PROJECT_ID or not LOCATION:
        raise ValueError("GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION must be set in .env")

    vertexai.init(project=PROJECT_ID, location=LOCATION)
    creds, _ = google.auth.default()

    llm = VertexAI(model_name="gemini-2.0-flash-001", credentials=creds)
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

# Retry wrapper for RAGAS evaluation
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

# Simple chunking logic
async def chunk_text(text, chunk_size, overlap):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Merge chunks while respecting max char length
async def merge_chunks_in_batches(chunks, max_char_len):
    merged_batches = []
    current_batch = ""
    current_len = 0

    for chunk in chunks:
        chunk_len = len(chunk)
        if current_len + chunk_len <= max_char_len:
            current_batch += " " + chunk
            current_len += chunk_len + 1
        else:
            merged_batches.append(current_batch.strip())
            current_batch = chunk
            current_len = chunk_len
    if current_batch:
        merged_batches.append(current_batch.strip())

    return merged_batches

# Used to create dataset from question, answer, context
async def createDataSet(merged_batches, question, answer, reference):
    data_samples = {
        'question': [],
        'answer': [],
        'contexts': [],
        'reference': []
    }
    for i, merged_context in enumerate(merged_batches):
        print(f"\nProcessing batch {i + 1}/{len(merged_batches)}")
        data_samples['question'].append(question)
        data_samples['answer'].append(answer)
        data_samples['contexts'].append([merged_context])
        data_samples['reference'].append(reference)
    
    return data_samples

# Run full evaluation pipeline
async def evaluate_dataset(data_samples) -> t.List[pd.DataFrame]:
    print("Initializing LLM and metrics...")
    wrapper, embeddings, metrics = get_llm_and_metrics()

    result_set = []
    dataset = Dataset.from_dict(data_samples)

    for i in range(len(dataset)):
        print(f"\n--- Evaluating Merged Chunk #{i + 1}/{len(dataset)} ---")
        result = evaluate_with_retries(i, dataset, wrapper, embeddings, metrics)
        if result is not None:
            result_set.append(result)

    if result_set:
        results_df = pd.concat(result_set)
        if "retrieved_contexts" in results_df.columns:
            results_df = results_df.drop(columns=["retrieved_contexts"])
    else:
        print("No successful evaluations.")

    return result_set