Feature: Sample Feature with Playwright and POM

  @smoke
  Scenario Outline: Verify homepage title test
    Given I open the homepage
    Then I should see the homepage title
    Then I evaulate response for "<question>" "<answer>" "<reference>" "<context_reference>"

    Examples:
      | question                     | answer                                                                                                                                                                      | reference                                                                                                                                                           | context_reference        |
      | Can lambdas return a value ? | Yes, lambdas can return a value. The type of the return value will be inferred by the compiler. The return statement is not required if the lambda body is just a one-liner | Lambdas may return a value. The type of the return value will be inferred by compiler. The return statement is not required if the lambda body is just a one-liner. | context_files/Java_8.pdf |
