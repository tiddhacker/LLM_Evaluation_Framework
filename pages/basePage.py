from playwright.async_api import Page

class BasePage:
    def __init__(self, page):
        self.page = page

    async def open_url(self, url):
        await self.page.goto(url)

    async def get_title(self):
        return await self.page.title()

    async def close(self):
        await self.page.close()
