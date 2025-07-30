from behave import given, then
from behave.api.async_step import async_run_until_complete
from pages.homePage import HomePage
from util.commonUtil import read_pdf

@given("I open the homepage")
@async_run_until_complete
async def open_homepage(context):
    context.page = await HomePage.create(context.page)
    await context.page.goToPage()

@then("I should see the homepage title")
@async_run_until_complete
async def verify_homepage_title(context):
    title = await context.page.get_homepage_title()
    print(f"Title: {title}")
    assert title and len(title) > 0

@then('I evaluate the LLM response for "{question}" "{answer}" "{reference}" "{context_reference}"')
@async_run_until_complete
async def evaluate_llm_response(context, question, answer, reference, context_reference):
    full_context = read_pdf(context_reference)
    result_set= await context.page.testLLM(full_context, question, answer, reference)
    # Store results to context.all_results. Will be used for generating final report in environment.py file
    if result_set:
        context.all_results.extend(result_set)

@then('I evaluate the LLM response with local model for "{question}" "{answer}" "{reference}"')
@async_run_until_complete
async def evaluate_llm_response(context, question, answer, reference):
    result_set= await context.page.testLLM_localModel(question, answer, reference)
    # Store results to context.all_results. Will be used for generating final report in environment.py file
    if result_set:
        context.all_results.extend(result_set)
