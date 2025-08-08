import json
import os
import pandas as pd
import typing as t

# Generate HTML and Excel report and publish to reports directory
async def generateEvaluationReport(report_name: str, result_set: t.List[pd.DataFrame]):
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

    html_table = results_df.to_html(index=False, table_id="resultsTable", border=0)

    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Evaluation Report</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
        <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/dataTables.bootstrap5.min.css">
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
                table-layout: fixed;
                border-collapse: separate;
                border-spacing: 10px;
                width: 100%;
                word-wrap: break-word;
            }}
            th, td {{
                text-align: left;
                vertical-align: top;
                padding: 8px;
                word-wrap: break-word;
                white-space: normal;
            }}
            thead th {{
                background-color: #007bff !important;
                color: #ffffff;
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
            $(document).ready(function() {{
                $('#resultsTable').DataTable({{
                    scrollX: true,
                    fixedHeader: true
                }});
            }});

            const ctx = document.getElementById('summaryChart').getContext('2d');
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


#to generate report for multi-model
def generate_html_report(df, output_file="evaluation_report.html"):
    # Simple inline CSS for table styling
    styles = """
    <style>
      body { font-family: Arial, sans-serif; margin: 20px; }
      h1 { color: #333; }
      table { border-collapse: collapse; width: 100%; margin-top: 20px; }
      th, td { border: 1px solid #ddd; padding: 8px; vertical-align: top; }
      th { background-color: #f2f2f2; text-align: left; }
      tr:nth-child(even) { background-color: #fafafa; }
      .metrics-key { font-weight: bold; }
      .metrics-comments { margin-top: 5px; font-style: italic; color: #555; }
    </style>
    """

    # HTML Header and Table Header
    html = f"""
    <html>
    <head><title>LLM Evaluation Report</title>{styles}</head>
    <body>
    <h1>LLM Evaluation Report</h1>
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>Question</th>
          <th>Expected Answer</th>
          <th>LLM Response</th>
          <th>Similarity Score</th>
          <th>Reasoning Score</th>
          <th>Metrics</th>
        </tr>
      </thead>
      <tbody>
    """

    for idx, row in df.iterrows():
        question = str(row["question"]).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        expected = str(row["expected_answer"]).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        response = str(row["llm_response"]).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        similarity_score = row.get("similarity_score", 0)
        reasoning_score = row.get("reasoning_score", 0)

        metrics_html = ""
        metrics_json_str = row.get("metrics_json", "{}")
        try:
            metrics_json = json.loads(metrics_json_str)
        except Exception:
            metrics_json = {}

        # Build nicely formatted metrics
        for key, value in metrics_json.items():
            if key == "comments":
                # Put comments in italic below the scores
                metrics_html += f'<div class="metrics-comments">{value}</div>'
            else:
                metrics_html += f'<div><span class="metrics-key">{key.capitalize()}:</span> {value}</div>'

        html += f"""
        <tr>
          <td>{idx + 1}</td>
          <td>{question}</td>
          <td>{expected}</td>
          <td>{response}</td>
          <td>{similarity_score:.2f}</td>
          <td>{reasoning_score:.2f}</td>
          <td>{metrics_html}</td>
        </tr>
        """

    html += """
      </tbody>
    </table>
    </body>
    </html>
    """

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
