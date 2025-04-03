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


====
In chatGPT search "how to create vertex api in google and set path for GOOGLE_APPLICATION_CREDENTIALS step by step"

Reference:
 ==========
Step 1: Set Up Google Cloud Project
Go to Google Cloud Console: Open the Google Cloud Console at https://console.cloud.google.com/.

Create a new project (if you don't already have one):

Click the project dropdown at the top of the console.

Click New Project.

Enter the project name and billing account.

Click Create.

Step 2: Enable Vertex AI API
Navigate to APIs & Services:

In the Google Cloud Console, go to the API & Services section from the left navigation panel.

Click Library.

Search for Vertex AI:

In the search bar, type Vertex AI.

Click on Vertex AI API.

Click the Enable button to activate the Vertex AI API.

Step 3: Create Service Account and Key
Go to IAM & Admin:

In the Google Cloud Console, go to IAM & Admin.

Click on Service accounts.

Create Service Account:

Click Create Service Account.

Enter a name and description for the service account.

Click Create.

Assign Roles:

For the service account, you can assign the necessary roles, such as:

Vertex AI User

Vertex AI Admin (if you need more permissions)

Click Continue.

Create Key:

After creating the service account, click on the three dots next to the service account you created.

Click Manage keys.

Click Add Key and select Create new key.

Choose JSON as the key type and click Create.

The key file (in .json format) will be downloaded to your computer.

Step 4: Set the GOOGLE_APPLICATION_CREDENTIALS Environment Variable
Find the path to your key file:

Locate the .json key file you downloaded in the previous step.

Set the GOOGLE_APPLICATION_CREDENTIALS environment variable:

If you're using Linux or macOS, open the terminal and run the following command:

bash
Copy
Edit
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your-service-account-file.json"
If you're using Windows, run the following command in Command Prompt (CMD):

cmd
Copy
Edit
set GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\your-service-account-file.json
For PowerShell, use:

powershell
Copy
Edit
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\your-service-account-file.json"
Verify the environment variable:

You can check if the environment variable is set correctly by running the following command:

bash
Copy
Edit
echo $GOOGLE_APPLICATION_CREDENTIALS