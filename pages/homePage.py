from .basePage import BasePage
from llmtesteval import evaluate_dataset, chunk_text, merge_chunks_in_batches, createDataSet
from util.commonUtil import *
from util.reportGen import *
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


    async def testLLM(self, full_context, question, answer, reference):
        # Step 1: Chunk the full context
        chunks = chunk_text(full_context, chunk_size=1000, overlap=200)

        # Step 2: Merge into batches within max character limit
        merged_batches = merge_chunks_in_batches(chunks, max_char_len=25000)

        print(f"Total Batches Created: {len(merged_batches)}")

        # Step 3: Create dataset
        data_samples = createDataSet(merged_batches, question, answer, reference)

        # Step 4: Evaluate all batches
        start_time = time.time()
        result_set = evaluate_dataset(data_samples)
        elapsed = time.time() - start_time
        print(f"\nEvaluation completed in {elapsed:.2f} seconds")

        # Step 5: Generate full report
        generateEvaluationReport("testReport", result_set)
