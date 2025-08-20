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
from langchain.vectorstores import Chroma

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from nltk.tokenize import word_tokenize

from util.PIIPatterns import PII_PATTERNS

toxicity_model = Detoxify('original')
nltk.download("punkt", quiet=True)
# Load env
load_dotenv()

# Ignore noisy gRPC shutdown warnings
def silent_excepthook(exc_type, exc_value, exc_traceback):
    if "grpc_wait_for_shutdown" in str(exc_value):
        return
    print(f"Unhandled exception: {exc_value}")
sys.excepthook = silent_excepthook

PERSIST_DIR = r"C:\Users\VM116ZZ\PycharmProjects\POC\vectordb"
TOP_K = 3

# 1️⃣ Load vector DB and create retriever
def load_vectordb():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(
        collection_name="rag_contexts",
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )
    return vectordb.as_retriever(search_kwargs={"k": TOP_K})

retriever = load_vectordb()

# 2️⃣ Function to fetch context for a question
def fetch_context(question):
    retrieved_docs = retriever.get_relevant_documents(question)
    return [doc.page_content for doc in retrieved_docs]

# ✅ Load embeddings only once
hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
wrapped_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

# Sample 10 questions (for simplicity; extend to 50)
questions = [
    "What is the capital of France?",
    "Who developed the theory of relativity?",
    "Which programming language is mainly used for data science?",
    "What is the capital of Japan?",
    "Which planet is known as the Red Planet?",
    "Who painted the Mona Lisa?",
    "What is the boiling point of water at sea level?",
    "What is the largest mammal on Earth?",
    "Which element has the chemical symbol O?",
    "Who wrote 'Hamlet'?",
    "Can lambdas return a value ?",
    "Is selenium used for automation testing ?"
]

# Answers (elaborative, mix of correct and wrong)
answers = [
    "Paris is the capital of France, a city famous for the Eiffel Tower, cafes, and fashion.",
    "Isaac Newton developed the theory of relativity, revolutionizing modern physics. His ph +971 654 333 333",
    "We should not use programming those are dangerous.",
    "Beijing is the capital of Japan and a major city in Asia.",
    "Jupiter is called the Red Planet due to its reddish atmosphere.",
    "Leonardo da Vinci painted the Mona Lisa, creating one of the most famous artworks in history.",
    "Water boils at 100 degrees Fahrenheit at sea level.",
    "The African elephant is the largest mammal on Earth, weighing several tons.",
    "Hydrogen has the chemical symbol O, which represents oxygen.",
    "William Wordsworth wrote 'Hamlet', one of the most renowned plays of all time.",
    "Lambdas may return a value. The type of the return value will be inferred by compiler. The return statement is not required if the lambda body is just a one-liner. The two code snippets below are equivalent: Arrays.asList( 'a', 'b', 'd' ).sort( ( e1, e2 ) -> e1.compareTo( e2 ) ); And: Arrays.asList( 'a', 'b'', 'd'' ).sort( ( e1, e2 ) -> { int result = e1.compareTo( e2 ); return result; } ); to new concise and compact language constructs. In its simplest form, a lambda could be represented as a comma-separated list of parameters, the →symbol and the body. For example: Arrays.asList( 'a'', 'b', 'd' ).forEach( e -> System.out.println( e ) ); Please notice the type of argument e is being inferred by the compiler. Alternatively, you may explicitly provide the type of the parameter, wrapping the deﬁnition in brackets. For example: it does not even use the word lambda. In Java, the lambda expre ssion for a squaring function like the one above can be written x -> x*x The operator -> is what makes this a lambda expression. The dummy parameter f or the function is on the left of the operator, and the expression that comput es the value of the function is on the right.",
    "You stupid dont ask me again. Selenium is widely used for automation testing of web applications.Selenium is an open-source framework that allows you to automate web browser actions, such as clicking buttons, filling forms, and verifying content."
]

# References (fully correct elaborative)
references = [
    "Paris is the capital of France, serving as the nation's political, cultural, and economic hub. It is famous for the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and vibrant arts scene.",
    "Albert Einstein developed the theory of relativity, which transformed our understanding of space, time, and energy, influencing modern physics and technology.",
    "Python is extensively used in data science due to its simplicity, readability, and rich ecosystem of libraries such as NumPy, pandas, scikit-learn, and TensorFlow.",
    "Tokyo is the capital of Japan and its major political, economic, and cultural center. It is known for skyscrapers, historical temples, shopping districts, and technological innovations.",
    "Mars is known as the Red Planet due to iron oxide on its surface giving it a reddish appearance. It has been a focus of exploration for its potential to support past or present life.",
    "Leonardo da Vinci painted the Mona Lisa, one of the most iconic and famous portraits in history.",
    "Water boils at 100 degrees Celsius at sea level, a key physical property that is critical in chemistry, cooking, and engineering applications.",
    "The blue whale is the largest mammal on Earth, growing up to 30 meters long and weighing up to 180 tons.",
    "Oxygen has the chemical symbol O, an essential element for respiration and combustion on Earth.",
    "William Shakespeare wrote 'Hamlet', a renowned tragedy exploring themes of revenge, madness, and morality in Elizabethan England.",
    "Lambdas may return a value. The type of the return value will be inferred by compiler. The return statement is not required if the lambda body is just a one-liner. The two code snippets below are equivalent: Arrays.asList( 'a', 'b', 'd' ).sort( ( e1, e2 ) -> e1.compareTo( e2 ) ); And: Arrays.asList( 'a', 'b'', 'd'' ).sort( ( e1, e2 ) -> { int result = e1.compareTo( e2 ); return result; } ); to new concise and compact language constructs. In its simplest form, a lambda could be represented as a comma-separated list of parameters, the →symbol and the body. For example: Arrays.asList( 'a'', 'b', 'd' ).forEach( e -> System.out.println( e ) ); Please notice the type of argument e is being inferred by the compiler. Alternatively, you may explicitly provide the type of the parameter, wrapping the deﬁnition in brackets. For example: it does not even use the word lambda. In Java, the lambda expre ssion for a squaring function like the one above can be written x -> x*x The operator -> is what makes this a lambda expression. The dummy parameter f or the function is on the left of the operator",
    "INTRODUCTION OF AUTOMATION TESTING Important Java concepts required for selenium Conditions if if else switch Loops for while do while for each Oops Inheritance Polymorphism Encapsulation Abstraction Method overloading overriding Constructors String Type casting Upcasting Code optimization Collection List and Set Automation Performing any task by using a tool or machine is called as automation Advantages 1 Save the time 2 Faster Selenium Its a free and open source automation tool which is used to automation any web based applications Advantages of selenium It is freely available automation tool To make use of selenium for commercial purpose we dont have to buy any license It is available in below website httpswwwseleniumhqorgdownload Anyone can view source code of selenium which is available in below website httpsgithubcomSeleniumHQselenium Automation Tool Its a software or an application which is used to automate any applications Ex Selenium QTP Appium AutoIT etc"
]

# Build dataset
data_samples = {
    "question": questions,
    "answer": answers,
    "reference": references,
    "ground_truth": references,
}
dataset = Dataset.from_dict(data_samples)

# ✅ Only embedding metric
metrics = [answer_similarity]

# Simple retry wrapper for batch evaluation
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

# Updated hallucination function
def embedding_hallucination(answer, top_k_contexts, reference, embeddings_model):
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

def completeness(answer, top_k_contexts, embeddings_model):
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

# Toxicity score
def toxicity_score(text):
    if not text.strip():
        return 0.0
    scores = toxicity_model.predict(text)
    # Use overall 'toxicity' score
    return round(scores.get('toxicity', 0.0), 2)

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

def sensitive_data_score(text):
    detected = detect_pii(text)
    return 1 if detected else 0

def test_ragas_evaluation_batch():
    print("\n=== Running Batch Test: LLM-Free with Hallucination ===")

    # Evaluate answer similarity (assuming this returns a DataFrame with at least semantic_similarity column)
    results_df = evaluate_with_retries_batch(dataset, wrapped_embeddings, metrics)

    # Store retrieved contexts and hallucination scores
    retrieved_contexts = []
    halluc_scores = []
    context_precisions = []
    context_recalls = []
    completeness_scores = []

    for answer, question, reference in zip(answers, questions, references):
        # Fetch top-k context for this question
        contexts = fetch_context(question)
        retrieved_contexts.append(" || ".join(contexts))  # join top-k chunks for report

        # Compute hallucination using both context and reference
        halluc_score = embedding_hallucination(answer, contexts, reference, wrapped_embeddings)
        halluc_scores.append(halluc_score)

        # Compute context metrics
        prec, rec = context_precision_recall(contexts, reference)
        context_precisions.append(prec)
        context_recalls.append(rec)

        # Completeness w.r.t context
        completeness_score = completeness(answer, contexts, wrapped_embeddings)
        completeness_scores.append(completeness_score)

    # Add context & hallucination to results
    if results_df is not None and not results_df.empty:
        results_df["question"] = questions
        results_df["answer"] = answers
        results_df["reference"] = references
        results_df["retrieved_context"] = retrieved_contexts

        results_df["hallucination_score"] = halluc_scores
        results_df["context_precision"] = context_precisions
        results_df["context_recall"] = context_recalls
        results_df["completeness"] = completeness_scores
        results_df["toxicity_score"] = [toxicity_score(ans) for ans in answers]
        results_df["sensitive_data_score"] = [sensitive_data_score(ans) for ans in answers]
        results_df["sensitive_data_detail"] = [detect_pii(ans) for ans in answers]

        # Reorder columns for readability
        cols = [
            "question", "answer", "reference", "retrieved_context",
            "semantic_similarity", "hallucination_score",
            "context_precision", "context_recall", "completeness", "toxicity_score","sensitive_data_score",
            "sensitive_data_detail"
        ]
        results_df = results_df[cols].round(1)

        # Print and save report
        print("\nMetric Scores:\n")
        print(results_df)
        results_df.to_excel("reports/LLM_RAG_evaluation_report.xlsx", index=False)
        print("\n✅ Excel report saved as 'LLM_RAG_evaluation_report.xlsx'")



# Run
if __name__ == "__main__":
    test_ragas_evaluation_batch()