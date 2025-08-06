import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import sys
import asyncio
import pandas as pd
import torch
import grpc
from dotenv import load_dotenv
from datasets import Dataset
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_chroma import Chroma
from ragas.evaluation import evaluate
from ragas.metrics import answer_similarity

def silent_excepthook(exc_type, exc_value, exc_traceback):
    print(f"Unhandled exception: {exc_value}")
sys.excepthook = silent_excepthook

load_dotenv()

# --- Load top-K retriever from Chroma -------------------------
PERSIST_DIR = r"C:\Users\VM116ZZ\PycharmProjects\POC\vectordb"
TOP_K = 3
def load_vectordb():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(
        collection_name="rag_contexts",
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )
    return vectordb.as_retriever(search_kwargs={"k": TOP_K})
# --------------------------------------------------------------


class LazyModelLoader:
    def __init__(self):
        self.model_loaded = False
        self.ragas_llm = None
        self.embeddings = None

    def load(self):
        if self.model_loaded:
            return

        # Loading local HuggingFace model
        print("Loading local model for evaluation...")
        local_model_name = os.getenv("MODEL_NAME")
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        local_model_path = os.path.join(BASE_DIR, "models", local_model_name)
        print("Model Directory:", local_model_path)

        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map="auto",
            attn_implementation="eager"
        )

        llm_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.3,
            do_sample=True
        )

        self.ragas_llm = HuggingFacePipeline(pipeline=llm_pipeline)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        print("Device set to use cpu")
        self.model_loaded = True


# Global model loader
lazy_loader = LazyModelLoader()
metrics = [answer_similarity]


async def evaluate_with_retries(index: int, dataset: Dataset, max_retries=3, delay=5):
    retries = 0
    while retries < max_retries:
        try:
            lazy_loader.load()  # Make sure model is loaded

            selected_data = dataset.select([index])
            result = evaluate(
                selected_data,
                metrics=metrics,
                llm=lazy_loader.ragas_llm,
                embeddings=lazy_loader.embeddings
            ).to_pandas()
            return result
        except grpc.RpcError as e:
            print(f"gRPC Error: {e}. Retrying {retries + 1}/{max_retries}...")
            await asyncio.sleep(delay * (2 ** retries))
            retries += 1
        except Exception as e:
            print(f"Unexpected Error in evaluate_with_retries(): {e}")
            return None
    print("Evaluation failed after retries.")
    return None


async def evaluate_dataset_localModel(question: str, answer: str, reference: str) -> pd.DataFrame | None:
    """
    Retrieve top-k contexts from Chroma and evaluate using local HuggingFace model.
    Returns a single Pandas DataFrame (or None).
    """
    # Step 1: retrieve relevant chunks
    retriever = load_vectordb()
    retrieved_docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in retrieved_docs]
    sources = [doc.metadata.get("source", "") for doc in retrieved_docs]

    print(f"\nTop-{TOP_K} retrieved context chunks for question: '{question}'")
    for idx, doc in enumerate(retrieved_docs, start=1):
        print(f"  [{idx}] Source: {doc.metadata.get('source', '')}")
        print(f"      Chunk: {doc.page_content[:150]}...\n")

    # Step 2: build dataset for ragas
    data_samples = {
        "question":     [question],
        "answer":       [answer],
        "contexts":     [contexts],
        "reference":    [reference],
        "ground_truth": [reference],
        "sources":      [sources],
    }

    results_set = []
    dataset = Dataset.from_dict(data_samples)

    # Step 3: evaluate
    print("\n--- Evaluating Answer Similarity (Local Model) ---")
    for i in range(len(dataset)):
        print(f"--- Evaluating sample #{i + 1}/{len(dataset)} ---")
        result = await evaluate_with_retries(i, dataset)
        if result is not None:
            results_set.append(result)

    if results_set:
        results_df = pd.concat(results_set)
        if "retrieved_contexts" in results_df.columns:
            results_df = results_df.drop(columns=["retrieved_contexts"])
    else:
        print("No evaluation completed.")

    return results_set