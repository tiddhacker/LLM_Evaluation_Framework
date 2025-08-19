PII_PATTERNS = {
    "email": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    # Phone: international code optional, digits in 2-5 blocks separated by space/dash/dot
    "phone": r"(?:\+?\d{1,3}[\s\-\.]?)?(?:\d{2,5}[\s\-\.]?){2,5}\d{2,5}",
    "ssn_us": r"\b\d{3}-\d{2}-\d{4}\b",
    "aadhaar": r"\b\d{4}-\d{4}-\d{4}\b|\b\d{12}\b",
    "pan": r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
    "bank_account": r"(?:\b(?:account|acc|iban|bank)\b[:\s]*)?(\d[\d\s\-]{7,19}\d)",
    "password": r"\b(?:password|pwd|pass)\b[:\s]*[^\s,;]+",
    "dob": r"\b\d{2}[\/\-]\d{2}[\/\-]\d{4}\b",
    "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    "location_coordinates": r"\b-?\d{1,3}\.\d+,\s*-?\d{1,3}\.\d+\b",
    "credit_card": r"\b(?:\d[ -]*?){13,16}\b",  # Visa, MasterCard, AmEx
    "insurance_number": r"\b(?:Insurance|Policy|ID|Pol|POL ID)[:\s]*[A-Z0-9\-]+\b",
}