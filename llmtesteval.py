import os
import time
import vertexai
from vertexai.language_models import TextGenerationModel
from dotenv import load_dotenv
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval import evaluate
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from datasets import Dataset 
from ragas import evaluate
from ragas.llms.base import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, answer_similarity
import pandas as pd
import google.auth
import grpc
import sys

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
ragas_vertexai_llm = VertexAI(model_name="gemini-pro", credentials=creds, request_timeout=600)
wrapper = LangchainLLMWrapper(ragas_vertexai_llm)

# Increase gRPC timeout
grpc_options = [("grpc.max_receive_message_length", 256 * 1024 * 1024),  # 256MB
                ("grpc.max_send_message_length", 256 * 1024 * 1024),
                ("grpc.keepalive_timeout_ms", 600000)]  # 10 min timeout

def create_secure_channel():
    return grpc.insecure_channel("vertexai.googleapis.com", options=grpc_options)

# Use custom RAGAS-compatible Vertex AI Embeddings
class RAGASVertexAIEmbeddings(VertexAIEmbeddings):
    async def embed_text(self, text: str) -> list[float]:
        return self.embed([text], 1, "SEMANTIC_SIMILARITY")[0]
    
    def set_run_config(self, *args, **kwargs):
        pass  # Dummy method to prevent errors

embeddings = RAGASVertexAIEmbeddings(model_name="text-embedding-005", credentials=creds)

# Define dataset
data_samples = {
    'question': ['When was the first super bowl?'],
    'answer': ['The first Super Bowl was held on January 15, 1967'],
    'contexts': [
        ['The First AFL–NFL World Championship Game was played on January 15, 1967, at the Los Angeles Memorial Coliseum.']
    ],
    'reference': ['Super Bowl I took place on January 15, 1967.']
}
dataset = Dataset.from_dict(data_samples)

# Initialize RAGAS Metrics
metrics = [
    answer_relevancy,
    context_recall,
    context_precision,
    answer_similarity,
]

for m in metrics:
    m.__setattr__("llm", wrapper)
    if hasattr(m, "embeddings"):
        m.__setattr__("embeddings", embeddings)

# Reset sys.excepthook to prevent issues
sys.excepthook = sys.__excepthook__

# Function to handle retries
def evaluate_with_retries(data, max_retries=3, delay=5):
    retries = 0
    while retries < max_retries:
        try:
            return evaluate(
                dataset.select(range(data, data + 1)),  # Process one entry at a time
                metrics=metrics,
                llm=wrapper,
                embeddings=embeddings
            ).to_pandas()
        except grpc.RpcError as e:
            print(f"gRPC Error: {e}. Retrying {retries+1}/{max_retries}...")
            time.sleep(delay * (2 ** retries))  # Exponential backoff
            retries += 1
        except Exception as e:
            print(f"Unexpected Error: {e}")
            sys.exit(1)  # Exit safely to prevent excepthook issues
    print("Evaluation failed after retries.")
    return None

# Run evaluation sequentially to prevent quota errors
result_set = []
for i in range(len(dataset)):
    result = evaluate_with_retries(i)
    if result is not None:
        result_set.append(result)

# Close gRPC channel after evaluation
channel = create_secure_channel()
channel.close()

# View results in a readable tabular format
if result_set:
    results_df = pd.concat(result_set)
    print("\nEvaluation Results:\n")
    print(results_df.to_markdown(index=False))  # Print as a well-formatted table
    




    
    # Generate HTML report dynamically based on user input
    report_name = input("Enter the report filename (without extension): ")
    report_filename = f"{report_name}.html"
    
    html_report = results_df.to_html(classes="table table-striped", index=False)
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Evaluation Report</title>
        <link rel="stylesheet" 
              href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    </head>
    <body class="container">
        <h2 class="mt-4 mb-4">Evaluation Results</h2>
        {html_report}
    </body>
    </html>
    """

    with open(report_filename, "w", encoding="utf-8") as file:
        file.write(html_template)
    print(f"\nEvaluation report generated: '{report_filename}'")
else:
    print("No successful evaluations.")
