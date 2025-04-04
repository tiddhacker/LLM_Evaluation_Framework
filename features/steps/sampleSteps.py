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


# Use async_run_until_complete to properly handle async steps
@then('I verify the factual correctness')
@async_run_until_complete
async def step_impl(context):
     # Load and concatenate context PDFs
    question= "Can lambdas return a value ?"
    answer="Yes, lambdas return a value";
    reference="Lambdas may return a value. The type of the return value will be inferred by compiler. The return statement is not required if the lambda body is just a one-liner. "
    context1 = read_pdf("context_files/Java_8.pdf")
    # context2 = read_pdf("context_files/javanotes8.pdf")
    full_context = context1
    await context.page.testLLM(full_context,question,answer,reference)
