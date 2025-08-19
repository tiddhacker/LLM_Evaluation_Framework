PII_PATTERNS = {
    "email": r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b",

    # Phone: require at least 7 digits and context for certainty
    "phone": r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?){1,3}\d{3,4}\b",

    "ssn_us": r"\b\d{3}-\d{2}-\d{4}\b",

    "aadhaar": r"\b\d{4}-\d{4}-\d{4}\b|\b\d{12}\b",

    "pan": r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",

    # Bank account requires explicit keywords for context
    "bank_account": r"\b(?:account|acc|iban|bank)\b[:\s]*\d{8,20}\b",

    # Password: require explicit "password"/"pwd"/"pass" before the string
    "password": r"\b(?:password|pwd|pass)\b[:\s]*[^\s,;]+",

    # Date of birth: allow DD/MM/YYYY or DD-MM-YYYY
    "dob": r"\b\d{2}[\/\-]\d{2}[\/\-]\d{4}\b",

    # IP address
    "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",

    # Location coordinates (lat,long)
    "location_coordinates": r"\b-?\d{1,3}\.\d+,\s*-?\d{1,3}\.\d+\b",

    # Credit card: require 13-16 digit sequences, optionally with spaces or dashes
    "credit_card": r"\b(?:\d[ -]*?){13,16}\b",

    # Insurance: require keywords
    "insurance_number": r"\b(?:insurance|policy|pol id)[:\s]*[A-Z0-9\-]+\b",
}