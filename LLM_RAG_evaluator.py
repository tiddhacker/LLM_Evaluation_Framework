import logging
import os
import sys
import pandas as pd
from datasets import Dataset

from ragas.metrics import answer_similarity
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_huggingface import HuggingFaceEmbeddings

from util.evaluation_metrics import toxicity_score, detect_pii, sensitive_data_score, embedding_hallucination_RAG, \
    context_precision_recall, completeness_RAG, evaluate_with_retries_batch, factual_consistency_score
from util.reportGen import html_report_LLM_RAG_evaluator
from util.vectorDB_util import load_vectordb

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
#=================Initialize Vector DB=============================
#==================================================================

retriever = load_vectordb()
def fetch_context(question):
    retrieved_docs = retriever.get_relevant_documents(question)
    return [doc.page_content for doc in retrieved_docs]

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

def test_ragas_evaluation_with_context():
    print("\n=== Running Batch Test: RAG Evaluation (With Retrieved Context) ===")
    results_df = evaluate_with_retries_batch(dataset, wrapped_embeddings, metrics)

    if results_df is not None and not results_df.empty:
        # --- Retrieved context ---
        retrieved_contexts = [ " || ".join(fetch_context(q)) for q in questions ]

        # --- Metrics ---
        halluc_scores = [
            embedding_hallucination_RAG(ans, fetch_context(q), ref, wrapped_embeddings)
            for ans, q, ref in zip(answers, questions, references)
        ]

        completeness_scores = [
            completeness_RAG(ans, fetch_context(q), wrapped_embeddings)
            for ans, q in zip(answers, questions)
        ]

        context_precisions, context_recalls = [], []
        for q, ref in zip(questions, references):
            prec, rec = context_precision_recall(fetch_context(q), ref)
            context_precisions.append(prec)
            context_recalls.append(rec)

        # --- Base Q/A/Ref/Context block ---
        base_df = pd.DataFrame({
            "question": questions,
            "answer": answers,
            "reference": references,
            "retrieved_context": retrieved_contexts
        })

        # --- Metrics block ---
        metrics_df = pd.DataFrame({
            "semantic_similarity": results_df["semantic_similarity"].round(2),
            "factual_consistency": [factual_consistency_score(ans, ref) for ans, ref in zip(answers, references)],
            "hallucination": [round(x, 2) for x in halluc_scores],
            "completeness": [round(x, 2) for x in completeness_scores],
            "context_precision": [round(x, 2) for x in context_precisions],
            "context_recall": [round(x, 2) for x in context_recalls],
            "toxicity_score": [round(float(toxicity_score(ans)), 2) for ans in answers],
            "sensitive_data_score": [round(sensitive_data_score(ans), 2) for ans in answers],
            "sensitive_data_detail": [detect_pii(ans) for ans in answers]
        })

        return base_df, metrics_df

    return None, None

# ==================================================================
# ================Run Evaluation & Report Gen=======================
# ==================================================================
base_df, metrics_df = test_ragas_evaluation_with_context()

if base_df is not None:
    final_df = pd.concat([base_df, metrics_df], axis=1)

    logging.info("\n Metric Scores:\n")
    logging.info("\n" + final_df.to_string())

    #generate excel report
    final_df.to_excel("reports/LLM_RAG_evaluation_report.xlsx", index=False)
    logging.info("\nExcel report saved as 'LLM_RAG_evaluation_report.xlsx'")

    #generate HTML report
    html_report_LLM_RAG_evaluator(final_df)

else:
    logging.info("\nNo evaluation results generated!")