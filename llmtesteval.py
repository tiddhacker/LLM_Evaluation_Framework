import os
import time
import vertexai
from vertexai.language_models import TextGenerationModel
from dotenv import load_dotenv
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval import evaluate as deepeval_evaluate
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
ragas_vertexai_llm = VertexAI(model_name="gemini-pro", credentials=creds, request_timeout=600)
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

# Report generator
def generateEvaluationReport(report_name, result_set):
    if not result_set:
        print("No data to generate report.")
        return

    results_df = pd.concat(result_set)
    if "retrieved_contexts" in results_df.columns:
        results_df = results_df.drop(columns=["retrieved_contexts"])



    # === HTML Report ===
    
def generateEvaluationReport(report_name: str, result_set: t.List[pd.DataFrame]):
    """
    Generates and saves evaluation reports in both HTML and Excel formats under the "reports" directory.

    Args:
        report_name (str): The name of the report (used for file naming).
        result_set (t.List[pd.DataFrame]): A list of Pandas DataFrames containing the evaluation results.
    """

    # Create the 'reports' directory if it doesn't exist
    reports_dir = "reports"
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)

    if not result_set:
        print("No data to generate report.")
        return

    results_df = pd.concat(result_set)
    if "retrieved_contexts" in results_df.columns:
        results_df = results_df.drop(columns=["retrieved_contexts"])

    report_filename_html = os.path.join(reports_dir, f"{report_name}.html")
    report_filename_xlsx = os.path.join(reports_dir, f"{report_name}.xlsx")

    # HTML table (no style, just raw table to apply DataTables enhancements)
    html_table = results_df.to_html(index=False, table_id="resultsTable", border=0)

    # === HTML Template with Styling, DataTables, Chart.js ===
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Evaluation Report</title>
        <link rel="stylesheet" 
              href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
        <link rel="stylesheet" 
              href="https://cdn.datatables.net/1.13.6/css/dataTables.bootstrap5.min.css">
        <style>
            body {{
                background-color: #f8f9fa;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }}
            h2 {{
                text-align: center;
                margin-top: 40px;
                margin-bottom: 30px;
                color: #343a40;
            }}
            table {{
                font-size: 0.95rem;
            }}
            thead th {{
                background-color: #007bff !important;
                color: #ffffff;
                text-align: center;
                vertical-align: middle;
            }}
            tbody td {{
                text-align: center;
                vertical-align: middle;
            }}
            .container {{
                max-width: 95%;
                margin: auto;
            }}
            canvas {{
                margin-bottom: 30px;
            }}
        </style>
    </head>
    <body class="container">
        <h2>Evaluation Results</h2>

        <canvas id="summaryChart" height="100"></canvas>

        {html_table}

        <script src="https://cdn.jsdelivr.net/npm/jquery@3.7.1/dist/jquery.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
        <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
        <script src="https://cdn.datatables.net/1.13.6/js/dataTables.bootstrap5.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>

        <script>
            // DataTables initialization
            $(document).ready(function() {{
                $('#resultsTable').DataTable({{
                    scrollX: true,
                    fixedHeader: true
                }});
            }});

            // Chart.js Bar Chart from numeric columns
            const ctx = document.getElementById('summaryChart').getContext('2d');
            const labels = {list(results_df.columns)};
            const numericData = {results_df.select_dtypes(include='number').mean().round(3).to_dict()};
            
            new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: Object.keys(numericData),
                    datasets: [{{
                        label: 'Average Scores',
                        data: Object.values(numericData),
                        backgroundColor: 'rgba(0, 123, 255, 0.7)',
                        borderColor: 'rgba(0, 123, 255, 1)',
                        borderWidth: 1
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        legend: {{ display: false }},
                        title: {{
                            display: true,
                            text: 'Average Metric Scores'
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            max: 1
                        }}
                    }}
                }}
            }});
        </script>
    </body>
    </html>
    """
    # Save HTML
    with open(report_filename_html, "w", encoding="utf-8") as file:
        file.write(html_template)
    print(f"✅ HTML report generated: '{report_filename_html}'")

    # Save Excel
    results_df.to_excel(report_filename_xlsx, index=False)
    print(f"✅ Excel report generated: '{report_filename_xlsx}'")