from playwright.sync_api import sync_playwright

def before_scenario(context, scenario):
    """Set up Playwright browser before each scenario."""

    context.playwright = sync_playwright().start()
    context.browser = context.playwright.chromium.launch(headless=False)
    context.page = context.browser.new_page()  # Ensure `page` is created

def after_scenario(context, scenario):
    """Teardown Playwright browser after each scenario."""
    if hasattr(context, "page"):  # Check if `page` exists before closing
        context.page.close()
    if hasattr(context, "browser"):
        context.browser.close()
    if hasattr(context, "playwright"):
        context.playwright.stop()
