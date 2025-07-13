import fitz  # PyMuPDF
import re

def clean_article_text(text):
    # Fix hyphenated line breaks: e.g., "multi-\npolar" → "multipolar"
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)

    # Fix mid-line hyphenated breaks: e.g., "multi- polar" → "multipolar"
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)

    # Replace common OCR or layout artifacts
    replacements = {
        "fve": "five",
        "frst": "first",
        "cofict": "conflict",
        "fnanciers": "financiers",
        "ofcial": "official",
        "Afairs": "Affairs",
        "fghting": "fighting",
        "afected": "affected",
        "fagged of": "flagged off",
        "safron ag": "saffron flag",
        "ash oods": "flash floods",
        "signi cant": "significant",
        "multipolar": "multipolar",
        "advantagevance": "advance",
        "To-ophobicbago": "Trinidad and Tobago",
        "Na-ophobicmibia": "Namibia",
    }

    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)

    # Remove odd non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # Remove known noisy headers
    text = re.sub(r'^\s*(A IN-X|YK|INSIDE|PAGE \d+|NEW DELHI|SRINAGAR|CHENNAI|KOLKATA)\s*$', '', text, flags=re.MULTILINE)

    # Normalize multiple spaces
    text = re.sub(r'[ \t]{2,}', ' ', text)

    # Separate potential articles
    text = re.sub(r'\n\s*\n+', '\n\n-----\n\n', text)

    return text.strip()

def extract_and_clean_all_pages(pdf_path, output_txt="all_pages_cleaned.txt"):
    with fitz.open(pdf_path) as doc:
        if len(doc) == 0:
            print("❌ PDF has no pages.")
            return

        full_text = ""
        for page_num, page in enumerate(doc):
            raw_text = page.get_text("text")
            cleaned = clean_article_text(raw_text)
            full_text += f"\n\n----- PAGE {page_num + 1} -----\n\n{cleaned}"

        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(full_text.strip())

    print(f"✅ Cleaned text from all pages saved to: '{output_txt}'")

# === Usage ===
pdf_file = "sample.pdf"  # Change to your PDF
extract_and_clean_all_pages(pdf_file)
