import re
import time
import nltk
from dotenv import load_dotenv
from detoxify import Detoxify
from nltk import word_tokenize

from ragas import evaluate

from sklearn.metrics.pairwise import cosine_similarity

from util.PIIPatterns import PII_PATTERNS

toxicity_model = Detoxify('original')
nltk.download("punkt", download_dir="resources/nltk_data")
nltk.download("punkt_tab", download_dir="resources/nltk_data")
nltk.download("stopwords", download_dir="resources/nltk_data")
nltk.download("wordnet", download_dir="resources/nltk_data")

#look for NLTK in below location
nltk.data.path.append("resources/nltk_data")

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
def embedding_hallucination_RAG(answer, top_k_contexts, reference, embeddings_model):
    """
    Compute hallucination score of an answer w.r.t. top-k context + reference embeddings.
    Higher similarity → lower hallucination.
    """
    # Combine context chunks and reference into one list
    grounding_texts = top_k_contexts + [reference]

    # Embed the answer
    answer_emb = embeddings_model.embed_documents([answer])[0]

    # Embed all grounding texts
    grounding_embs = embeddings_model.embed_documents(grounding_texts)

    # Compute cosine similarities between answer and each grounding embedding
    sims = [cosine_similarity([answer_emb], [g_emb])[0][0] for g_emb in grounding_embs]

    # Average similarity
    avg_sim = sum(sims) / len(sims) if sims else 0.0

    # Hallucination score: 1 - average similarity
    halluc_score = 1 - avg_sim
    return halluc_score


def context_precision_recall(context, reference):
    # Simple word overlap
    context_tokens = set(word_tokenize(" ".join(context).lower()))
    reference_tokens = set(word_tokenize(reference.lower()))

    true_positives = len(context_tokens & reference_tokens)

    precision = true_positives / len(context_tokens) if context_tokens else 1.0
    recall = true_positives / len(reference_tokens) if reference_tokens else 1.0

    return precision, recall

def completeness_RAG(answer, top_k_contexts, embeddings_model):
    """
    Computes semantic completeness score of answer w.r.t top-k retrieved context using embeddings.
    Returns a 0-1 score.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    if not top_k_contexts:
        return 1.0  # empty context => consider complete

    # Concatenate context chunks
    context_text = " ".join(top_k_contexts)

    # Embed answer and context
    answer_emb = embeddings_model.embed_query(answer)
    context_emb = embeddings_model.embed_query(context_text)

    # Compute cosine similarity as completeness
    score = cosine_similarity([answer_emb], [context_emb])[0][0]
    return float(score)












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








