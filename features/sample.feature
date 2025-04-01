Feature: Sample Feature with Playwright and POM

  @smoke
  Scenario: Verify homepage title test
    Given I open the homepage
    Then I should see the homepage title
    Then I verify the factual correctness