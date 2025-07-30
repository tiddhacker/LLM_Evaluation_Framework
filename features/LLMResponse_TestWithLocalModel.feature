Feature: LLM Local model evaluation

  @localLLM
  Scenario Outline: Verify results with local model
    Given I open the homepage
    Then I should see the homepage title
    Then I evaluate the LLM response with local model for "<question>" "<answer>" "<reference>"

    Examples:
      | question                     | answer                                                                                                                                                                      | reference                                                                                                                                                                                                                                                                                                |
      | Can lambdas return a value ? | Yes, lambdas can return a value. The type of the return value will be inferred by the compiler. The return statement is not required if the lambda body is just a one-liner | Lambdas may return a value. The type of the return value will be inferred by compiler. The return statement is not required if the lambda body is just a one-liner.                                                                                                                                      |
      | How does python work ?       | Sorry this is out of my area of expertise.                                                                                                                                  | Python works by executing code line by line through an interpreter, which reads and translates the code directly into machine-executable instructions. This process, known as interpretation, allows for immediate execution without the need for explicit compilation into machine code before running. |
