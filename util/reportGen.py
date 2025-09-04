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

def generate_html_reportRag(df, output_file="evaluation_report.html"):
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
          <th>Context Precision Score</th>
          <th>Faithfulness Score</th>
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
        context_precision_score = row.get("context_precision_score", 0)
        faithfulness_score = row.get("faithfulness_score", 0)
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
          <td>{context_precision_score:.2f}</td>
          <td>{faithfulness_score:.2f}</td>
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


def html_report_LLM_evaluator(final_df):
    # Normalize column names
    final_df = final_df.rename(columns={
        "sensitive_data_detail": "Sensitive Data Detail",
        "sensitive_data_score": "Sensitive Data Score",
        "semantic_similarity": "Semantic Similarity",
        "hallucination": "Hallucination",
        "completeness": "Completeness",
        "toxicity_score": "Toxicity Score",
        "factual_consistency": "Factual Consistency"
    })

    # Replace {} or [] with blank
    if "Sensitive Data Detail" in final_df.columns:
        final_df["Sensitive Data Detail"] = final_df["Sensitive Data Detail"].apply(
            lambda x: "" if str(x).strip() in ["{}", "[]"] else str(x)
        )

    # Apply red/orange highlighting based on thresholds
    def highlight(val, col):
        try:
            v = float(val)
            if col == "Semantic Similarity":
                if v < 0.4: return 'class="poor"'
                elif v < 0.75: return 'class="medium"'
            elif col == "Factual Consistency":
                if v < 0.4: return 'class="poor"'
                elif v < 0.85: return 'class="medium"'
            elif col == "Hallucination":
                if v > 0.6: return 'class="poor"'
                elif v > 0.3: return 'class="medium"'
            elif col == "Completeness":
                if v < 0.4: return 'class="poor"'
                elif v < 0.7: return 'class="medium"'
            elif col == "Toxicity Score":
                if v > 0.6: return 'class="poor"'
                elif v > 0.2: return 'class="medium"'
            elif col=="Sensitive Data Score":
                if v > 0: return 'class="poor"'
        except:
            return ""
        return ""

    # Build HTML table manually with highlights + row numbers
    headers = ["#"] + final_df.columns.tolist()
    rows_html = ""
    for idx, row in final_df.iterrows():
        row_html = f"<tr><td>{idx+1}</td>"  # row numbers
        for col in final_df.columns:
            cell_val = row[col]
            attr = highlight(cell_val, col)
            row_html += f"<td {attr}>{cell_val}</td>"
        row_html += "</tr>"
        rows_html += row_html

    # HTML Template with modern CSS
    html_content = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, sans-serif;
                margin: 20px;
                background-color: #f8f9fa;
            }}
            h2 {{
                text-align: center;
                color: #333;
                margin-bottom: 20px;
            }}
            .legend {{
                margin-bottom: 20px;
                padding: 12px;
                border: 1px solid #ccc;
                border-radius: 8px;
                background: #ffffff;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}
            .table-container {{
                max-height: 600px;
                overflow-y: auto;
                overflow-x: auto;
                border: 1px solid #ddd;
                border-radius: 6px;
                background: #fff;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                table-layout: fixed;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px 10px;
                text-align: left;
                vertical-align: top;
                white-space: pre-wrap;
                word-wrap: break-word;
            }}
            th {{
                position: sticky;
                top: 0;
                background-color: #f2f4f7;
                color: #333;
                font-weight: bold;
                text-align: center;
                border-bottom: 2px solid #ddd;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            tr:hover {{
                background-color: #f1f7ff;
            }}
            .poor {{
                background-color: #ffcccc !important;
            }}
            .medium {{
                background-color: #ffd580 !important;
            }}
            #searchInput {{
                margin-bottom: 10px;
                padding: 10px;
                width: 100%;
                border: 1px solid #ccc;
                border-radius: 6px;
                font-size: 14px;
            }}
        </style>
        <script>
            function filterTable() {{
                var input, filter, table, tr, td, i, j, txtValue;
                input = document.getElementById("searchInput");
                filter = input.value.toLowerCase();
                table = document.getElementById("resultTable");
                tr = table.getElementsByTagName("tr");

                for (i = 1; i < tr.length; i++) {{
                    tr[i].style.display = "none";
                    td = tr[i].getElementsByTagName("td");
                    for (j = 0; j < td.length; j++) {{
                        if (td[j]) {{
                            txtValue = td[j].textContent || td[j].innerText;
                            if (txtValue.toLowerCase().indexOf(filter) > -1) {{
                                tr[i].style.display = "";
                                break;
                            }}
                        }}
                    }}
                }}
            }}
        </script>
    </head>
    <body>
        <h2>📊 LLM Evaluation Report</h2>

        <div class="legend">
        <h2>Legend:</h2>
        <ul>
            <li><b>Semantic Similarity:</b> Measures closeness in meaning to reference. Higher is better (0.8+ good).</li>
            <li><b>Factual Consistency:</b> Measures the factual data in answers comparing it to reference. Higher is better (0.85+ good).</li>
            <li><b>Hallucination:</b> Checks if answer invents unsupported info. Lower is better (&lt;0.3 good).</li>
            <li><b>Completeness:</b> Measures how fully answer covers reference. Higher is better (0.7+ good).</li>
            <li><b>Toxicity Score:</b> Detects offensive/unsafe language. Lower is better (&lt;0.2 safe).</li>
            <li><b>Sensitive Data Score:</b> Detects PII like Aadhaar, phone, bank info. 1 means PII detected.</li>
        </ul>
        </div>


        <input type="text" id="searchInput" onkeyup="filterTable()" placeholder="🔍 Search records...">

        <div class="table-container">
            <table id="resultTable">
                <thead>
                    <tr>{"".join(f"<th>{h}</th>" for h in headers)}</tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """

    with open("reports/LLM_evaluation_report.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    print("\nHTML report saved as 'LLM_evaluation_report.html'")

def html_report_LLM_RAG_evaluator(final_df):
    # Normalize column names
    final_df = final_df.rename(columns={
        "sensitive_data_detail": "Sensitive Data Detail",
        "sensitive_data_score": "Sensitive Data Score",
        "semantic_similarity": "Semantic Similarity",
        "hallucination": "Hallucination",
        "completeness": "Completeness",
        "toxicity_score": "Toxicity Score",
        "context_precision": "Context Precision",
        "context_recall": "Context Recall",

    })

    #excluding column retrieved_context
    if "retrieved_context" in final_df.columns:
        final_df = final_df.drop(columns=["retrieved_context"])

    # Replace {} or [] with blank
    if "Sensitive Data Detail" in final_df.columns:
        final_df["Sensitive Data Detail"] = final_df["Sensitive Data Detail"].apply(
            lambda x: "" if str(x).strip() in ["{}", "[]"] else str(x)
        )

    # Apply red/orange highlighting based on thresholds
    def highlight(val, col):
        try:
            v = float(val)
            if col == "Semantic Similarity":
                if v < 0.4: return 'class="poor"'
                elif v < 0.7: return 'class="medium"'
            elif col == "Hallucination":
                if v > 0.6: return 'class="poor"'
                elif v > 0.3: return 'class="medium"'
            elif col == "Completeness":
                if v < 0.4: return 'class="poor"'
                elif v < 0.7: return 'class="medium"'
            elif col == "Toxicity Score":
                if v > 0.6: return 'class="poor"'
                elif v > 0.2: return 'class="medium"'
            elif col == "Context Precision":
                if v < 0.4: return 'class="poor"'
                elif v < 0.7: return 'class="medium"'
            elif col == "Context Recall":
                if v < 0.4: return 'class="poor"'
                elif v < 0.7: return 'class="medium"'
            elif col=="Sensitive Data Score":
                if v > 0: return 'class="poor"'
        except:
            return ""
        return ""

    # Build HTML table manually with highlights + row numbers
    headers = ["#"] + final_df.columns.tolist()
    rows_html = ""
    for idx, row in final_df.iterrows():
        row_html = f"<tr><td>{idx+1}</td>"  # row numbers
        for col in final_df.columns:
            cell_val = row[col]
            attr = highlight(cell_val, col)
            row_html += f"<td {attr}>{cell_val}</td>"
        row_html += "</tr>"
        rows_html += row_html

    # HTML Template with modern CSS
    html_content = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, sans-serif;
                margin: 20px;
                background-color: #f8f9fa;
            }}
            h2 {{
                text-align: center;
                color: #333;
                margin-bottom: 20px;
            }}
            .legend {{
                margin-bottom: 20px;
                padding: 12px;
                border: 1px solid #ccc;
                border-radius: 8px;
                background: #ffffff;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}
            .table-container {{
                max-height: 600px;
                overflow-y: auto;
                overflow-x: auto;
                border: 1px solid #ddd;
                border-radius: 6px;
                background: #fff;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                table-layout: fixed;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px 10px;
                text-align: left;
                vertical-align: top;
                white-space: pre-wrap;
                word-wrap: break-word;
            }}
            th {{
                position: sticky;
                top: 0;
                background-color: #f2f4f7;
                color: #333;
                font-weight: bold;
                text-align: center;
                border-bottom: 2px solid #ddd;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            tr:hover {{
                background-color: #f1f7ff;
            }}
            .poor {{
                background-color: #ffcccc !important;
            }}
            .medium {{
                background-color: #ffd580 !important;
            }}
            #searchInput {{
                margin-bottom: 10px;
                padding: 10px;
                width: 100%;
                border: 1px solid #ccc;
                border-radius: 6px;
                font-size: 14px;
            }}
        </style>
        <script>
            function filterTable() {{
                var input, filter, table, tr, td, i, j, txtValue;
                input = document.getElementById("searchInput");
                filter = input.value.toLowerCase();
                table = document.getElementById("resultTable");
                tr = table.getElementsByTagName("tr");

                for (i = 1; i < tr.length; i++) {{
                    tr[i].style.display = "none";
                    td = tr[i].getElementsByTagName("td");
                    for (j = 0; j < td.length; j++) {{
                        if (td[j]) {{
                            txtValue = td[j].textContent || td[j].innerText;
                            if (txtValue.toLowerCase().indexOf(filter) > -1) {{
                                tr[i].style.display = "";
                                break;
                            }}
                        }}
                    }}
                }}
            }}
        </script>
    </head>
    <body>
        <h2>📊 LLM Evaluation Report</h2>

        <div class="legend">
        <h2>Legend:</h2>
        <ul>
            <li><b>Semantic Similarity:</b> Measures closeness in meaning to reference. Higher is better (0.7+ good).</li>
            <li><b>Hallucination:</b> Checks if answer invents unsupported info (w.r.t. top-k context + reference). Lower is better (&lt;0.3 good).</li>
            <li><b>Completeness:</b> Measures how fully answer covers reference. Higher is better (0.7+ good).</li>
            <li><b>Context Precision:</b> Measures how much of the retrieved context is actually relevant to the reference. Higher is better (0.7+ good).</li>
            <li><b>Context Recall:</b> Measures how much of the reference is covered by the retrieved context. Higher is better (0.7+ good).</li>
            <li><b>Toxicity Score:</b> Detects offensive/unsafe language. Lower is better (&lt;0.2 safe).</li>
            <li><b>Sensitive Data Score:</b> Detects PII like Aadhaar, phone, bank info. 1 means PII detected.</li>
        </ul>
        </div>


        <input type="text" id="searchInput" onkeyup="filterTable()" placeholder="🔍 Search records...">

        <div class="table-container">
            <table id="resultTable">
                <thead>
                    <tr>{"".join(f"<th>{h}</th>" for h in headers)}</tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """

    with open("reports/LLM_RAG_evaluation_report.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    print("\nHTML report saved as 'LLM_RAG_evaluation_report.html'")

