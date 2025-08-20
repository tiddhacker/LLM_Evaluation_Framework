import re
import time
import sys
import nltk
from dotenv import load_dotenv
from datasets import Dataset
from detoxify import Detoxify

from ragas import evaluate
from ragas.metrics import answer_similarity
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

from util.PIIPatterns import PII_PATTERNS

toxicity_model = Detoxify('original')
nltk.download("punkt", quiet=True)
load_dotenv()

#==================================================================
#===========COMMON METRICS DEF FOR LLM AND RAG RESPONSES===========
#==================================================================

# Toxicity score
def toxicity_score(text):
    if not text.strip():
        return 0.0
    scores = toxicity_model.predict(text)
    # Use overall 'toxicity' score
    return round(scores.get('toxicity', 0.0), 2)

#IDENTIFY PII
COMPILED_PATTERNS = {k: re.compile(v, re.IGNORECASE) for k, v in PII_PATTERNS.items()}
def detect_pii(text):
    detected = {}
    for key, pattern in COMPILED_PATTERNS.items():
        matches = re.findall(pattern, text)
        # For numeric sequences, require context words to avoid false positives
        if key in ["bank_account", "credit_card", "insurance_number", "password"]:
            if matches:
                detected[key] = matches
        else:
            if matches:
                detected[key] = matches
    return detected

#PII Scoring based on above IDENTIFIED PII
def sensitive_data_score(text):
    detected = detect_pii(text)
    return 1 if detected else 0


#==================================================================
#===================METRICS DEF FOR LLM RESPONSES==================
#==================================================================

# Hallucination (answer vs reference)
def embedding_hallucination(answer, reference, embeddings_model):
    answer_emb = embeddings_model.embed_documents([answer])[0]
    reference_emb = embeddings_model.embed_documents([reference])[0]
    sim = cosine_similarity([answer_emb], [reference_emb])[0][0]
    halluc_score = 1 - sim
    return halluc_score


def semantic_completeness_score(answer, reference, embeddings_model):
    """
    Computes a completeness score (0-1) based on semantic similarity of answer vs reference.
    """
    from nltk.tokenize import sent_tokenize
    # Split into sentences
    ref_sents = sent_tokenize(reference)
    ans_sents = sent_tokenize(answer)

    if not ref_sents:
        return 1.0

    # Embed all sentences
    ref_embs = embeddings_model.embed_documents(ref_sents)
    ans_embs = embeddings_model.embed_documents(ans_sents)

    # For each reference sentence, find max similarity in answer sentences
    sims = []
    for r_emb in ref_embs:
        max_sim = max(cosine_similarity([r_emb], ans_embs)[0])
        sims.append(max_sim)

    # Average over all reference sentences
    return sum(sims) / len(sims)

#==================================================================
#================METRICS DEF FOR RAG LLM RESPONSES=================
#==================================================================













#==================================================================
#================Evaluation method=================================
#==================================================================
def evaluate_with_retries_batch(dataset, embeddings, metrics, max_retries=3, delay=5):
    retries = 0
    while retries < max_retries:
        try:
            result = evaluate(
                dataset,
                metrics=metrics,
                embeddings=embeddings
            ).to_pandas()
            return result
        except Exception as e:
            print(f"Error during evaluation: {e}. Retrying {retries + 1}/{max_retries}...")
            time.sleep(delay * (2 ** retries))
            retries += 1
    print("Evaluation failed after retries.")
    return None








