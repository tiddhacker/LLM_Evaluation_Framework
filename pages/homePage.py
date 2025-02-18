from .basePage import BasePage
from llmEvaluatorUtil import *

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
        sampleFinal = createSampleDataSet("This is Java", "This is Python")
        score = await fetch_score_with_retry(sampleFinal)