from .basePage import BasePage
from llmEvaluatorUtil import *
from llmtesteval import evaluate_dataset, generateEvaluationReport
from util.commonUtil import *

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

    async def testLLM(self,full_context,question,answer,reference):

        # Chunk the combined context
        chunks = self.chunk_text(full_context, chunk_size=1000, overlap=200)

        # Create data samples per chunk
        data_samples = {
            'question': [],
            'answer': [],
            'contexts': [],
            'reference': []
        }

        # Evaluate top N chunks (e.g., 5)
        for chunk in chunks:
            data_samples['question'].append(question)
            data_samples['answer'].append(answer)
            data_samples['contexts'].append([chunk])
            data_samples['reference'].append(reference)

        # Run evaluation
        result_set = evaluate_dataset(data_samples)
        generateEvaluationReport("testReport", result_set)
