import fitz  # PyMuPDF

def read_pdf(file_path):
    """Reads text from a PDF file and returns it as a string."""
    text = ""
    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text("text") + "\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None
    return text
