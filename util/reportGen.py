import os
import pandas as pd
from torch import t
import typing as t


#generate html and excel report and publish to reports directory
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