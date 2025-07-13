import fitz  # PyMuPDF
import re
import os

def clean_article_text(text):
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)

    replacements = {
        "fve": "five", "frst": "first", "cofict": "conflict", "fnanciers": "financiers",
        "ofcial": "official", "Afairs": "Affairs", "fghting": "fighting", "afected": "affected",
        "fagged of": "flagged off", "safron ag": "saffron flag", "ash oods": "flash floods",
        "signi cant": "significant", "multipolar": "multipolar", "advantagevance": "advance",
        "To-ophobicbago": "Trinidad and Tobago", "Na-ophobicmibia": "Namibia"
    }
    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)

    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'^\s*(A IN-X|YK|INSIDE|PAGE \d+|NEW DELHI|SRINAGAR|CHENNAI|KOLKATA)\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n\n-----\n\n', text)
    return text.strip()

def split_pdf_to_chunks(pdf_path, pages_per_chunk=2, output_dir="chunks"):
    os.makedirs(output_dir, exist_ok=True)
    with fitz.open(pdf_path) as doc:
        chunk_index = 1
        for i in range(0, len(doc), pages_per_chunk):
            chunk_text = ""
            for j in range(i, min(i + pages_per_chunk, len(doc))):
                raw_text = doc[j].get_text("text")
                cleaned = clean_article_text(raw_text)
                chunk_text += f"\n\n----- PAGE {j + 1} -----\n\n{cleaned}\n"

            file_path = os.path.join(output_dir, f"chunk_{chunk_index}.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(chunk_text.strip())
            print(f"✅ Saved: {file_path}")
            chunk_index += 1

# === Usage ===
if __name__ == "__main__":
    pdf_file = "sample.pdf"  # Replace with your PDF filename
    split_pdf_to_chunks(pdf_file, pages_per_chunk=2)

    # === Summarizer Integration ===
    from summrizer import process_file, save_to_mongodb

    chunks_dir = "chunks"
    for filename in sorted(os.listdir(chunks_dir)):
        if filename.endswith(".txt"):
            file_path = os.path.join(chunks_dir, filename)
            print(f"🧠 Summarizing {file_path}...")
            summaries = process_file(file_path)
            save_to_mongodb(summaries, input_path=file_path)
            print(f"✅ Summarized and saved: {filename}")
