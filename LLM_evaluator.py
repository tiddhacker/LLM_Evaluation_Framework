import os
import sys
import logging

import pandas as pd
from datasets import Dataset

from ragas.metrics import answer_similarity
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_huggingface import HuggingFaceEmbeddings

from util.evaluation_metrics import embedding_hallucination, semantic_completeness_score, toxicity_score, \
    sensitive_data_score, detect_pii, evaluate_with_retries_batch
from util.reportGen import html_report_LLM_evaluator

logging.basicConfig(level=logging.INFO, format="%(message)s")


#==================================================================
#=============Ignore noisy gRPC shutdown warnings==================
#==================================================================
def silent_excepthook(exc_type, exc_value, exc_traceback):
    if "grpc_wait_for_shutdown" in str(exc_value):
        return
    print(f"Unhandled exception: {exc_value}")
sys.excepthook = silent_excepthook

#==================================================================
#====================LOAD EMBEDDINGS===============================
#==================================================================
embeddings = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL_NAME"))
wrapped_embeddings = LangchainEmbeddingsWrapper(embeddings)


#==================================================================
#====================TEST DATA SAMPLE==============================
#==================================================================

# Build dataset
df = pd.read_excel("resources/TestData/LLM_evaluator_dataset.xlsx")
dataset = Dataset.from_pandas(df)
# Extract columns
questions = df["question"].tolist()
answers = df["answer"].tolist()
references = df["reference"].tolist()


#==================================================================
#====================Evaluation:Test Method========================
#==================================================================
metrics = [answer_similarity]
def test_ragas_evaluation_batch():
    print("\n=== Running Batch Test: LLM-Free (No Context) ===")
    results_df = evaluate_with_retries_batch(dataset, wrapped_embeddings, metrics)

    halluc_scores = [
        embedding_hallucination(ans, ref, wrapped_embeddings)
        for ans, ref in zip(answers, references)
    ]

    completeness_score = [
        semantic_completeness_score(ans, ref, wrapped_embeddings)
        for ans, ref in zip(answers, references)
    ]

    if results_df is not None and not results_df.empty:
        # --- Base Q/A/Ref block ---
        base_df = pd.DataFrame({
            "question": questions,
            "answer": answers,
            "reference": references
        })

        # --- Metrics block ---
        metrics_df = pd.DataFrame({
            "semantic_similarity": results_df["semantic_similarity"].round(2),
            "hallucination": [round(x, 2) for x in halluc_scores],
            "completeness": [round(x, 2) for x in completeness_score],
            "toxicity_score": [round(float(toxicity_score(ans)), 2) for ans in answers],
            "sensitive_data_score": [round(sensitive_data_score(ans), 2) for ans in answers],
            "sensitive_data_detail": [detect_pii(ans) for ans in answers]
        })

        return base_df, metrics_df

    return None, None

# ==================================================================
# ================Run Evaluation & Report Gen=======================
# ==================================================================
base_df, metrics_df = test_ragas_evaluation_batch()

if base_df is not None:
    final_df = pd.concat([base_df, metrics_df], axis=1)

    logging.info("\n Metric Scores:\n")
    logging.info("\n" + final_df.to_string())

    #generate excel report
    final_df.to_excel("reports/LLM_evaluation_report.xlsx", index=False)
    logging.info("\nExcel report saved as 'LLM_evaluation_report.xlsx'")

    #generate HTML report
    html_report_LLM_evaluator(final_df)

else:
    logging.info("\nNo evaluation results generated!")
