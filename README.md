#Run the scripts using-
1. Install VS Code (It supports mapping of feature vs step def, which pycharm won't support in community edition)
2. Install cucumber, python extension in vs code
3. Create Vertex API KEY and place the file under resources/API_KEYS/VertexAPIKey
4. In .env file give the file name under (VERTEX_APIKEY_FILE_NAME)
4. In .env file update project id and location. Project id to be taken from json key file
3. Open terminal
4. Run behave --tags=smoke

#For setting up local model-
1. In env file just keep LOCAL MODEL DETAILS

MODEL_NAME= give the model name

2. comment out all other lines


====

Metric	Description
====================
answer_relevancy:	How well the answer addresses the question.
answer_similarity:	Similarity between generated and reference answer.
context_precision:	Proportion of retrieved context that is relevant.
context_recall: 	Proportion of relevant context that was retrieved.
faithfulness:		How factually consistent the answer is with the retrieved context.
answer_correctness:	Checks if the answer is semantically correct compared to the reference.

====


Search "how to create and download vertex api key in json format step by step"

Reference:
===
Set Up Google Cloud Project
Go to Google Cloud Console: Open the Google Cloud Console at https://console.cloud.google.com/.


====
To run local llm download offline model from here -

https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/tree/main

===
