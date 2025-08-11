# import os
# os.environ["TRANSFORMERS_NO_TF"] = "1"
#
# import sys
# import asyncio
# import typing as t
# import pandas as pd
# import fitz
# import torch
# import grpc
#
# from dotenv import load_dotenv
# from datasets import Dataset
# from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
# from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
# from ragas.evaluation import evaluate
# from ragas.metrics import (
#     answer_relevancy,
#     context_precision,
#     context_recall,
#     answer_similarity,
#     faithfulness,
# )
# from util.reportGen import generateEvaluationReport
#
# def silent_excepthook(exc_type, exc_value, exc_traceback):
#     print(f"Unhandled exception: {exc_value}")
# sys.excepthook = silent_excepthook
#
# load_dotenv()
#
# # Load HF model
# local_model_path = "models/phi-3-mini-4k-instruct"
#
# # Load tokenizer and model from local path
# tokenizer = AutoTokenizer.from_pretrained(local_model_path)
# model = AutoModelForCausalLM.from_pretrained(
#     local_model_path,
#     trust_remote_code=True,
#     torch_dtype=torch.float32,
#     device_map="auto",
#     attn_implementation="eager"
# )
#
# llm_pipeline = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=512,
#     temperature=0.3,
#     do_sample=True  # fixes warning about `temperature` without sampling
# )
# ragas_llm = HuggingFacePipeline(pipeline=llm_pipeline)
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#
# metrics = [
#     answer_relevancy,
#     context_precision,
#     context_recall,
#     answer_similarity,
#     faithfulness
# ]
#
# async def evaluate_with_retries(index: int, dataset: Dataset, max_retries=3, delay=5):
#     retries = 0
#     while retries < max_retries:
#         try:
#             selected_data = dataset.select([index])
#             result = evaluate(
#                 selected_data,
#                 metrics=metrics,
#                 llm=ragas_llm,
#                 embeddings=embeddings
#             ).to_pandas()
#             return result
#         except grpc.RpcError as e:
#             print(f"gRPC Error: {e}. Retrying {retries + 1}/{max_retries}...")
#             await asyncio.sleep(delay * (2 ** retries))
#             retries += 1
#         except Exception as e:
#             print(f"Unexpected Error in evaluate_with_retries(): {e}")
#             import traceback
#             traceback.print_exc()
#             return None
#     print("Evaluation failed after retries.")
#     return None
#
# async def chunk_text(text, chunk_size, overlap):
#     words = text.split()
#     chunks = []
#     for i in range(0, len(words), chunk_size - overlap):
#         chunk = " ".join(words[i:i + chunk_size])
#         chunks.append(chunk)
#     return chunks
#
# async def merge_chunks_in_batches(chunks, max_char_len):
#     merged_batches = []
#     current_batch = ""
#     current_len = 0
#     for chunk in chunks:
#         chunk_len = len(chunk)
#         if current_len + chunk_len <= max_char_len:
#             current_batch += " " + chunk
#             current_len += chunk_len + 1
#         else:
#             merged_batches.append(current_batch.strip())
#             current_batch = chunk
#             current_len = chunk_len
#     if current_batch:
#         merged_batches.append(current_batch.strip())
#     return merged_batches
#
# async def createDataSet(merged_batches, question, answer, reference):
#     return {
#         'question': [question] * len(merged_batches),
#         'answer': [answer] * len(merged_batches),
#         'contexts': [[ctx] for ctx in merged_batches],
#         'reference': [reference] * len(merged_batches)
#     }
#
# async def evaluate_dataset(data_samples) -> t.List[pd.DataFrame]:
#     result_set = []
#     dataset = Dataset.from_dict(data_samples)
#     for i in range(len(dataset)):
#         print(f"\n--- Evaluating Merged Chunk #{i + 1}/{len(dataset)} ---")
#         result = await evaluate_with_retries(i, dataset)
#         if result is not None:
#             result_set.append(result)
#     if result_set:
#         results_df = pd.concat(result_set)
#         print("\nEvaluation Results:\n")
#         print(results_df.to_markdown(index=False))
#     else:
#         print("No successful evaluations.")
#     return result_set
#
# def read_pdf(path):
#     text = ""
#     with fitz.open(path) as doc:
#         for page in doc:
#             text += page.get_text()
#     return text
#
# async def main():
#     # context_path = "context_files/List-of-Presidents-of-India.pdf"
#     question = "Who is the president of India?"
#     answer = "As of May 2025, the President of India is Droupadi Murmu."
#     reference = "Sounak Ghosh"
#
#     # full_context = read_pdf(context_path)
#     full_context="Droupadi Murmu is the 15ᵗʰ President of India, sworn in on 25 July 2022. She is distinguished as the first person from a tribal community, the second woman (after Pratibha Patil), the youngest India President at the time of election, and the first born post-independence"
#     chunks = await chunk_text(full_context, chunk_size=150, overlap=20)
#     merged = await merge_chunks_in_batches(chunks, max_char_len=1000)
#     samples = await createDataSet(merged, question, answer, reference)
#     result_set = await evaluate_dataset(samples)
#     await generateEvaluationReport("testReport", result_set)
#
# if __name__ == "__main__":
#     asyncio.run(main())