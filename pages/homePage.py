from .basePage import BasePage
from RAGAS_GeminiEvaluator import evaluate_single_question
import time

class HomePage(BasePage):
    def __init__(self, page):
        super().__init__(page)

    @classmethod
    async def create(cls, page):
        return cls(page)

    async def goToPage(self):
        await self.open_url("https://www.google.com")

    async def get_homepage_title(self):
        return await self.get_title()

    async def close(self):
        await self.page.close()


    async def testLLM(self, question, answer, reference):
        start_time = time.time()
        result_set = await evaluate_single_question(question, answer, reference)
        elapsed = time.time() - start_time
        print(f"\nEvaluation completed in {elapsed:.2f} seconds")
        return result_set
        # Step 5: Generate full report
        # await generateEvaluationReport("testReport", result_set)


    # async def testLLM_localModel(self, question, answer, reference):
    #     # Step 1: Evaluate using local model
    #     start_time = time.time()
    #     result_set= await evaluate_dataset_localModel(question, answer, reference)
    #     elapsed = time.time() - start_time
    #     print(f"\nEvaluation completed in {elapsed:.2f} seconds")
    #     return result_set
    #
    #     # Step 2: Generate full report
    #     # await generateEvaluationReport("testReport", result_set)
