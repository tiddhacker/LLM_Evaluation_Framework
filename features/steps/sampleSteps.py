from behave import *
from behave.api.async_step import async_run_until_complete
from pages.homePage import HomePage

@given("I open the homepage")
@async_run_until_complete
async def open_homepage(context):
    context.page = HomePage(context.page)
    context.page.goToPage()

@then("I should see the homepage title")
@async_run_until_complete
async def verify_homepage_title(context):
    print(context.page.get_homepage_title())


# Use async_run_until_complete to properly handle async steps
@then('I verify the factual correctness')
@async_run_until_complete
async def step_impl(context):
    await context.page.testLLM()
