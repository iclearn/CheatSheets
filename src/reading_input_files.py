import fitz
from pdfminer.high_level import extract_text

def read_pdf(filepath):
    text = extract_text(filepath)
    doc = fitz.open(filepath)

def is