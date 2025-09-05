PII_PATTERNS = {
    "email": r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b",

    # Phone: allow "phone/mobile number is"
    "phone": r"\b(?:phone|mobile|tel|contact)(?:\s*number)?(?:\s*is|:)?\s*(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?){1,3}\d{3,4}\b",

    # SSN
    "ssn_us": r"\b(?:ssn|social\s*security)(?:\s*number|id)?(?:\s*is|:)?\s*\d{3}-\d{2}-\d{4}\b",

    # Aadhaar: accept "Aadhar" OR "Aadhaar"
    "aadhaar": r"\b(?:aadhar|aadhaar)(?:\s*number|id)?(?:\s*is|:)?\s*(?:\d{4}[- ]\d{4}[- ]\d{4}|\d{12})\b",

    "pan": r"\b(?:pan|permanent\s*account)(?:\s*number|id)?(?:\s*is|:)?\s*[A-Z]{5}[0-9]{4}[A-Z]\b",

    "bank_account": r"\b(?:account|acc|iban|bank\s*account)(?:\s*number|id)?(?:\s*is|:)?\s*\d{8,20}\b",

    "password": r"\b(?:password|pwd|pass)(?:\s*is|:)?\s*[^\s,;]+",

    "dob": r"\b(?:dob|date\s*of\s*birth)(?:\s*is|:)?\s*\d{2}[\/\-]\d{2}[\/\-]\d{4}\b",

    "ip_address": r"\b(?:ip|ip\s*address)(?:\s*is|:)?\s*(?:\d{1,3}\.){3}\d{1,3}\b",

    "location_coordinates": r"\b(?:location|coordinates|latlong)(?:\s*is|:)?\s*-?\d{1,3}\.\d+,\s*-?\d{1,3}\.\d+\b",

    "credit_card": r"\b(?:card|credit\s*card|debit\s*card|cc)(?:\s*number|id)?(?:\s*is|:)?\s*(?:\d[ -]*?){13,16}\b",

    "insurance_number": r"\b(?:insurance|policy)\s*(?:number|id)(?:\s*is|:)?\s*[A-Z0-9\-]{4,}\b",
}