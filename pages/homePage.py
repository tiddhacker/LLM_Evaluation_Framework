from .basePage import BasePage
from llmEvaluatorUtil import *
from llmtesteval import evaluate_dataset, generateEvaluationReport

class HomePage(BasePage):
    def __init__(self, page):
        super().__init__(page)  # Correctly initialize BasePage

    def goToPage(self):
        self.open_url("https://www.google.com")  # Calling BasePage method

    def get_homepage_title(self):
        return self.get_title()  # Calling BasePage method

    def close(self):
        self.page.close()  # Closing the browser page

    async def testLLM(self):
        # sampleFinal = createSampleDataSet("This is Java", "This is Java")
        # score = await fetch_score_with_retry(sampleFinal)
        data_samples = {
            'question': ['When was the first super bowl?'],
            'answer': ['The first Super Bowl was not held'],
            'contexts': [
              ['java is the best.']
            ],
            'reference': ['Super Bowl I took place on January 15, 1967.']
        }
        result_set=evaluate_dataset(data_samples)
        generateEvaluationReport("testReport",result_set)
