import os
import asyncio
import google.generativeai as genai
from dotenv import load_dotenv
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import FactualCorrectness
from ragas.llms import LangchainLLMWrapper
from langchain_google_genai import GoogleGenerativeAI

load_dotenv()

# Retrieve Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please set it in your environment.")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Set up Gemini LLM using LangChain wrapper
llm = GoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

# # Create a sample response and reference
# sample = SingleTurnSample(
#     response="The Eiffel Tower is located in Paris.",
#     reference="Java is best"
# )


def createSampleDataSet(res, ref):
    sample = SingleTurnSample(
        response=res,
        reference=ref
    )
    return sample

# sampleFinal= createSampleDataSet("This is Java", "This is Java")

# Initialize FactualCorrectness scorer with Gemini LLM
scorer = FactualCorrectness()
scorer.llm = LangchainLLMWrapper(llm)


# Retry function with backoff
async def fetch_score_with_retry(sample):
    max_retries = 5
    retries = 0
    while retries < max_retries:
        try:
            score = await scorer.single_turn_ascore(sample)
            print("Factual Correctness Score:", score)
            return score  # Exit the function after a successful request
        except Exception as e:
            retries += 1
            if retries < max_retries:
                wait_time = 2 ** retries  # Exponential backoff
                print(f"Retrying... Attempt {retries}/{max_retries} after {wait_time} seconds")
                await asyncio.sleep(wait_time)  # Wait before retrying
            else:
                print(f"Max retries reached. Error: {e}")
                raise e  # Raise the last error after exceeding retries


# # Run async function with retry
# async def main():
#     await fetch_score_with_retry()
#
#
# asyncio.run(main())  # Ensure async execution
