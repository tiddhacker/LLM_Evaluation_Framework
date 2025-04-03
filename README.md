#Run the scripts using-
1. Install VS Code (It supports mapping of feature vs step def, which pycharm won't support in community edition)
2. Install cucumber, python extension in vs code
3. Open terminal
4. Run behave --tags=smoke


====
Key Metrics in ragas:
----------------------

Factual Correctness – Ensures the generated response is factually correct.

Context Relevance – Measures how relevant the retrieved context is to the query.

Answer Relevance – Evaluates if the generated answer is relevant to the query.

Faithfulness – Checks if the answer is fully supported by the retrieved context.

Context Precision – Ensures the retrieved documents do not contain unnecessary information.

Answer Correctness – Evaluates correctness using a reference answer.
