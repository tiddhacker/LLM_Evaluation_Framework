import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import sys
import asyncio
import typing as t
import pandas as pd
import torch
import grpc

from dotenv import load_dotenv
from datasets import Dataset
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from ragas.evaluation import evaluate
from ragas.metrics import answer_similarity
from util.reportGen import generateEvaluationReport

def silent_excepthook(exc_type, exc_value, exc_traceback):
    print(f"Unhandled exception: {exc_value}")
sys.excepthook = silent_excepthook

load_dotenv()

class LazyModelLoader:
    def __init__(self):
        self.model_loaded = False
        self.ragas_llm = None
        self.embeddings = None

    def load(self):
        if self.model_loaded:
            return

        #loading local model
        print("Loading local model for evaluation...")
        local_model_name = os.getenv("MODEL_NAME")
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        local_model_path= os.path.join(BASE_DIR, "models", local_model_name)
        print("Model Directory: ",local_model_path)

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


# Global lazy loader instance
lazy_loader = LazyModelLoader()

# Only semantic similarity
metrics = [answer_similarity]


async def evaluate_with_retries(index: int, dataset: Dataset, max_retries=3, delay=5):
    retries = 0
    while retries < max_retries:
        try:
            lazy_loader.load()  # Ensure model is loaded before first eval

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
            import traceback
            traceback.print_exc()
            return None
    print("Evaluation failed after retries.")
    return None


def createDataSet(question: str, answer: str, reference: str):
    return {
        'question': [question],
        'answer': [answer],
        'contexts': [[""]],  # Required dummy context
        'reference': [reference]
    }


async def evaluate_dataset_localModel(question: str, answer: str, reference: str) -> t.List[pd.DataFrame]:
    data_samples = createDataSet(question, answer, reference)
    dataset = Dataset.from_dict(data_samples)
    print("\n--- Evaluating Answer Similarity ---")
    result = await evaluate_with_retries(0, dataset)
    if result is not None:
        print("\nEvaluation Result:\n")
        print(result.to_markdown(index=False))
        return [result]
    else:
        print("Evaluation failed.")
        return []


# async def main():
#     question = "Who is the president of India?"
#     answer = "As of May 2025, the President of America is Droupadi Murmu."
#     reference = "Sounak Ghosh is my name"

#     result_set = await evaluate_dataset_localModel(question, answer, reference)
#     await generateEvaluationReport("semantic_similarity_report", result_set)


# if __name__ == "__main__":
#     asyncio.run(main())