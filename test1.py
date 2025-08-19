import re

# Define PII patterns
PII_PATTERNS = {
    # Generic bank account: 8 to 20 digits, allow spaces or dashes, preceded by optional keywords
    "bank_account": r"(?:\b(?:account|acc|iban|bank)\b[:\s]*)?(\d[\d\s\-]{7,19}\d)"
}

# Compile regex patterns
COMPILED_PATTERNS = {k: re.compile(v, re.IGNORECASE) for k, v in PII_PATTERNS.items()}

# Function to calculate PII score
def calculate_pii_score(text):
    for name, pattern in COMPILED_PATTERNS.items():
        if pattern.search(text):
            print(f"PII detected: {name}")
            return 1.0
    return 0.0

# Runner function
def run_pii_detection(texts):
    results = []
    for t in texts:
        score = calculate_pii_score(t)
        results.append((t, score))
        print(f"Text: {t}\nPII Score: {score}\n{'-'*50}")
    return results

# Sample texts
sample_texts = [
    "World War II caused widespread destruction, millions of casualties, and significant geopolitical changes, including the emergence of the US and USSR as superpowers. Bank acc: 9876-5432-1098",
    "Marie Curie’s research in radioactivity led to the discovery of polonium and radium, pioneering studies in physics and chemistry, and two Nobel Prizes. My account number is 123456789012",
    "No PII here",
]

# Run detection if executed as a script
if __name__ == "__main__":
    run_pii_detection(sample_texts)
