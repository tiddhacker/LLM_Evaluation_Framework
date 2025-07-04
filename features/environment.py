from playwright.async_api import async_playwright
import asyncio
import nest_asyncio

nest_asyncio.apply()

def before_all(context):
    context.loop = asyncio.get_event_loop()
    context.playwright = context.loop.run_until_complete(async_playwright().start())
    context.browser = context.loop.run_until_complete(
        context.playwright.chromium.launch(headless=False)
    )

def before_scenario(context, scenario):
    context.browser_context = context.loop.run_until_complete(context.browser.new_context())
    context.page = context.loop.run_until_complete(context.browser_context.new_page())

def after_scenario(context, scenario):
    context.loop.run_until_complete(context.page.close())
    context.loop.run_until_complete(context.browser_context.close())

def after_all(context):
    context.loop.run_until_complete(context.browser.close())
    context.loop.run_until_complete(context.playwright.stop())