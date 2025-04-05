import os
import time
import vertexai
from dotenv import load_dotenv
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from datasets import Dataset
from ragas import evaluate
from ragas.llms.base import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, answer_similarity
import pandas as pd
import google.auth
import grpc
import sys
import typing as t

# Load environment variables
load_dotenv()
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")

if not PROJECT_ID or not LOCATION:
    raise ValueError("Google Cloud PROJECT_ID and LOCATION must be set in environment variables.")

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Load Gemini model via Vertex AI
creds, project_id = google.auth.default()
ragas_vertexai_llm = VertexAI(model_name="gemini-2.0-flash-001", credentials=creds, request_timeout=600)
wrapper = LangchainLLMWrapper(ragas_vertexai_llm)

# gRPC settings
grpc_options = [
    ("grpc.max_receive_message_length", 256 * 1024 * 1024),
    ("grpc.max_send_message_length", 256 * 1024 * 1024),
    ("grpc.keepalive_timeout_ms", 600000)
]

def create_secure_channel():
    return grpc.insecure_channel("vertexai.googleapis.com", options=grpc_options)

# Custom embeddings class
class RAGASVertexAIEmbeddings(VertexAIEmbeddings):
    async def embed_text(self, text: str) -> list[float]:
        return self.embed([text], 1, "SEMANTIC_SIMILARITY")[0]

    def set_run_config(self, *args, **kwargs):
        pass

embeddings = RAGASVertexAIEmbeddings(model_name="text-embedding-005", credentials=creds)

# Attach LLM and embeddings to metrics
metrics = [answer_relevancy, context_recall, context_precision, answer_similarity]
for m in metrics:
    m.llm = wrapper
    if hasattr(m, "embeddings"):
        m.embeddings = embeddings

# Reset excepthook
sys.excepthook = sys.__excepthook__

# Retry wrapper
def evaluate_with_retries(index: int, dataset: Dataset, max_retries=3, delay=5):
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
            print(f"gRPC Error: {e}. Retrying {retries+1}/{max_retries}...")
            time.sleep(delay * (2 ** retries))
            retries += 1
        except Exception as e:
            print(f"Unexpected Error: {e}")
            sys.exit(1)
    print("Evaluation failed after retries.")
    return None


def chunk_text(text, chunk_size, overlap):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def merge_chunks_in_batches(chunks, max_char_len):
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

def createDataSet(merged_batches,question, answer, reference):
    data_samples = {
        'question': [],
        'answer': [],
        'contexts': [],
        'reference': []
    }
    batch_scores = []
    for i, merged_context in enumerate(merged_batches):
        print(f"\nProcessing batch {i + 1}/{len(merged_batches)}")
        data_samples['question'].append(question)
        data_samples['answer'].append(answer)
        data_samples['contexts'].append([merged_context])
        data_samples['reference'].append(reference)
    
    return data_samples
        

# Dataset evaluator
def evaluate_dataset(data_samples) -> t.List[pd.DataFrame]:
    result_set = []
    dataset = Dataset.from_dict(data_samples)

    for i in range(len(dataset)):
        print(f"\n--- Evaluating Merged Chunk #{i+1} ---")
        # print(f"Question: {dataset[i]['question']}")
        # print(f"Answer: {dataset[i]['answer']}")
        # print(f"Reference: {dataset[i]['reference']}")
        # print(f"Context (First 300 chars): {dataset[i]['contexts'][0][:300]}...\n")

        result = evaluate_with_retries(i, dataset)
        if result is not None:
            result_set.append(result)

    channel = create_secure_channel()
    channel.close()

    if result_set:
        results_df = pd.concat(result_set)
        if "retrieved_contexts" in results_df.columns:
            results_df = results_df.drop(columns=["retrieved_contexts"])

        # To print result in terminal:
        # print("\nEvaluation Results:\n")
        # print(results_df.to_markdown(index=False))
    else:
        print("No successful evaluations.")

    return result_set