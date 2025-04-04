from .basePage import BasePage
from llmEvaluatorUtil import *
from llmtesteval import evaluate_dataset, generateEvaluationReport
from util.commonUtil import *
import time

class HomePage(BasePage):
    def __init__(self, page):
        super().__init__(page)

    def goToPage(self):
        self.open_url("https://www.google.com")

    def get_homepage_title(self):
        return self.get_title()

    def close(self):
        self.page.close()

    def chunk_text(self, text, chunk_size=1000, overlap=200):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    def merge_chunks_in_batches(self, chunks, max_char_len=25000):
        merged_batches = []
        current_batch = ""
        current_len = 0

        for chunk in chunks:
            chunk_len = len(chunk)
            if current_len + chunk_len <= max_char_len:
                current_batch += " " + chunk
                current_len += chunk_len + 1
            else:
                merged_batches.append(current_batch.strip())
                current_batch = chunk
                current_len = chunk_len
        if current_batch:
            merged_batches.append(current_batch.strip())

        return merged_batches

    async def testLLM(self, full_context, question, answer, reference):
        # Step 1: Chunk the full context
        chunks = self.chunk_text(full_context, chunk_size=1000, overlap=200)

        # Step 2: Merge into batches within max character limit
        merged_batches = self.merge_chunks_in_batches(chunks, max_char_len=25000)

        print(f"Total Batches Created: {len(merged_batches)}")

        # Step 3: Create dataset
        data_samples = {
            'question': [],
            'answer': [],
            'contexts': [],
            'reference': []
        }

        batch_scores = []
        for i, merged_context in enumerate(merged_batches):
            print(f"\nProcessing batch {i + 1}/{len(merged_batches)}")
            data_samples['question'].append(question)
            data_samples['answer'].append(answer)
            data_samples['contexts'].append([merged_context])
            data_samples['reference'].append(reference)

        # Step 4: Evaluate all batches
        start_time = time.time()
        result_set = evaluate_dataset(data_samples)
        elapsed = time.time() - start_time
        print(f"\nEvaluation completed in {elapsed:.2f} seconds")

        # Step 5: Generate full report
        generateEvaluationReport("testReport", result_set)
