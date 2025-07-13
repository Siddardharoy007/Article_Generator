
# 📄 PDF Summarizer with Clean Page Text Extraction

## 🛠️ Overview

This project extracts clean text from a PDF file and generates **bullet-point summaries with headings** for each article or section. It combines:
- `PyMuPDF (fitz)` for reading PDFs,
- `spaCy` for sentence tokenization,
- `facebook/bart-large-cnn` from Hugging Face for summarization.

---

## 📦 Requirements

Install the required packages:
```bash
pip install pymupdf spacy transformers torch
python -m spacy download en_core_web_sm
```

---

## 📁 Project Structure

| File                  | Purpose                                                                 |
|-----------------------|-------------------------------------------------------------------------|
| `text.py`             | Extracts text from each PDF page with clean 5-line page separation      |
| `summrizer.py`        | Splits content into articles, summarizes each, and generates headings   |
| `summary_output.txt`  | Final summary output (headings + bullet points)                         |
| `sample.pdf`          | Example PDF used for extraction                                         |
| `output.txt`          | Raw extracted text from PDF before summarization                        |
| `test_spacy.py`       | Tests entity recognition with spaCy                                     |
| `pratice_programs.py` | Extracts **only the first page** of a PDF into a text file              |

---

## 🧾 How It Works

### 🔹 Step 1: Extract Text per Page (`text.py`)
```bash
python text.py
```
- Loads the PDF (`sample.pdf`)
- Extracts text from each page
- Adds `--- Page X ---` headers and 5 newlines for readability
- Saves the result to `output.txt`

---

### 🔹 Step 2: Summarize Articles (`summrizer.py`)
```bash
python summrizer.py
```
- Reads `output.txt`
- Splits into separate articles based on paragraph gaps
- Summarizes each article using `facebook/bart-large-cnn`
- Adds a short generated heading for each summary
- Outputs to `summary_output.txt`

---

### 🔹 Step 3: View Output

The final result is saved in `summary_output.txt`. Each section looks like this:

```
📌 Heading: India and Brazil will discuss the priorities of the Global South
📄 Summary:
 - India and Brazil will discuss ways to advance the priorities of the Global South.
 - PM Modi will attend the BRICS summit and meet various leaders.
 - The trip includes visits to Brazil, Ghana, and more.
------------------------------------------------------------
```

---

## 🧪 Test (Optional)

You can test spaCy's NER model using:
```bash
python test_spacy.py
```

---

## 📝 Notes

- Works best for PDFs with **text-based** content (not scanned images).
- `facebook/bart-large-cnn` is a robust summarizer for news-like content.
- Avoid using overly short articles (<100 words) for meaningful summarization.

---

## 📈 Future Improvements

- Add OCR for image-based PDFs using Tesseract.
- Add keyword extraction or topic classification.
- Export results in JSON or CSV format.
