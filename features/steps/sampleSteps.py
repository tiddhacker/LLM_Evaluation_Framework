from behave import *
from behave.api.async_step import async_run_until_complete
from pages.homePage import HomePage
from util.commonUtil import *

@given("I open the homepage")
@async_run_until_complete
async def open_homepage(context):
    context.page = HomePage(context.page)
    context.page.goToPage()

@then("I should see the homepage title")
@async_run_until_complete
async def verify_homepage_title(context):
    print(context.page.get_homepage_title())

@then('I evaulate response for "{question}" "{answer}" "{reference}" "{context_reference}"')
@async_run_until_complete
async def step_impl(context, question,answer,reference,context_reference):
    # Load and concatenate context PDFs
    ques= question
    ans=answer
    ref=reference
    context1 = read_pdf(context_reference)
    # context2 = read_pdf("context_files/javanotes8.pdf")
    full_context = context1
    await context.page.testLLM(full_context,ques,ans,ref)