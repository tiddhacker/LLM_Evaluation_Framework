from playwright.async_api import async_playwright
import asyncio
import nest_asyncio

from util.reportGen import generateEvaluationReport

nest_asyncio.apply()

def before_all(context):
    context.all_results = []     # A list to collect all scenario result dataframes

    try:
        context.loop = asyncio.get_event_loop()
        context.playwright = context.loop.run_until_complete(async_playwright().start())
        context.browser = context.loop.run_until_complete(
            context.playwright.chromium.launch(headless=False,channel="chrome")
        )
    except Exception as e:
        print(f"[before_all] Error starting Playwright or browser: {e}")
        context.playwright = None
        context.browser = None


def before_scenario(context, scenario):
    if context.browser is None:
        raise RuntimeError("Browser was not initialized in before_all.")

    context.browser_context = context.loop.run_until_complete(context.browser.new_context())
    context.page = context.loop.run_until_complete(context.browser_context.new_page())


def after_scenario(context, scenario):
    if hasattr(context, "page"):
        context.loop.run_until_complete(context.page.close())
    if hasattr(context, "browser_context"):
        context.loop.run_until_complete(context.browser_context.close())


def after_all(context):
    import asyncio
    import pandas as pd

    async def _generate():
        # Only include actual DataFrames
        df_only = [r for r in context.all_results if isinstance(r, pd.DataFrame)]
        if df_only:
            await generateEvaluationReport("FinalEvaluationReport", df_only)
        else:
            print("No results to report.")

    asyncio.get_event_loop().run_until_complete(_generate())

    if hasattr(context, "browser") and context.browser:
        context.loop.run_until_complete(context.browser.close())
    if hasattr(context, "playwright") and context.playwright:
        context.loop.run_until_complete(context.playwright.stop())