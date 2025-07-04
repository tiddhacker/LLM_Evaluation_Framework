#Run the scripts using-
1. Install VS Code (It supports mapping of feature vs step def, which pycharm won't support in community edition)
2. Install cucumber, python extension in vs code
3. Set path for GOOGLE_APPLICATION_CREDENTIALS (Search "how to create vertex api in google and set path for GOOGLE_APPLICATION_CREDENTIALS)
3. Open terminal
4. Run behave --tags=smoke


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


Search "how to create vertex api in google and set path for GOOGLE_APPLICATION_CREDENTIALS step by step"

eg. - place json key in user directory and browse it to set path. 
variable name: GOOGLE_APPLICATION_CREDENTIALS
variable value: browse it to json file you got from google

Reference:
===
Set Up Google Cloud Project
Go to Google Cloud Console: Open the Google Cloud Console at https://console.cloud.google.com/.


====
To run local llm download offline model from here -

https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/tree/main

===