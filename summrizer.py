import re
import spacy
from transformers import pipeline, AutoTokenizer

# Load spaCy English model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    raise RuntimeError("Run: python -m spacy download en_core_web_sm")

# Load summarizer pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

# === Text Utilities ===
def spacy_sent_tokenize(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def clean_truncated_heading(heading):
    heading = re.sub(r'\b(?:in|of|at|for|by|on|with|during|and|to)\b$', '', heading.strip(), flags=re.IGNORECASE)
    heading = re.sub(r'\b\w{1,3}$', '', heading).strip()
    return heading

def generate_hashtags(text, max_tags=5):
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

    # Final stoplist for cleaning up unwanted hashtags
    tag_stoplist = set([
        "news", "centre", "end", "hear", "high", "allow", "government", "leader",
        "ajith", "ajithkumar", "ahuja", "horizontal", "france", "halt", "page", "let",
        "home", "body", "company", "crime", "expand", "batch", "lead", "cooperation",
        "facility", "international", "assembly", "balapur", "factory", "kill", "die",
        "bring", "day", "bow", "conical", "emblem", "ablakwa"
    ])

    # Extract heading and first summary sentences for relevance filtering
    preview = " ".join(spacy_sent_tokenize(text)[:4]).lower()

    # Preprocess text
    doc = nlp(text)
    cleaned = ' '.join([token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha])

    # TF-IDF extraction
    tfidf = TfidfVectorizer(max_features=100, stop_words='english')
    tfidf_matrix = tfidf.fit_transform([cleaned])
    feature_array = np.array(tfidf.get_feature_names_out())
    tfidf_scores = tfidf_matrix.toarray()[0]
    top_indices = tfidf_scores.argsort()[::-1]
    top_keywords = feature_array[top_indices][:max_tags * 3]

    # Named entities filtered by relevance
    entities = set()
    for ent in doc.ents:
        if ent.label_ in ["GPE", "ORG", "EVENT", "PERSON"]:
            ent_text = ent.text.strip().lower()
            if ent_text in preview and ent_text not in tag_stoplist:
                entities.add(ent_text.replace(' ', ''))

    # Filtered and cleaned TF-IDF terms
    keywords = set()
    for kw in top_keywords:
        kw_clean = kw.strip().lower().replace(' ', '')
        if kw_clean not in tag_stoplist and 2 < len(kw_clean) < 25:
            keywords.add(kw_clean)

    keywords.update(entities)

    hashtags = [f"#{kw}" for kw in sorted(keywords)]
    return hashtags[:max_tags]

# === Split into Articles ===
def split_into_articles(text):
    sections = re.split(r'\n-{3,}\n', text.strip())
    return [s.strip() for s in sections if len(s.strip()) > 100 and not s.strip().startswith("PAGE ")]

# === Generate Heading ===
def generate_heading(article_text):
    sentences = spacy_sent_tokenize(article_text)
    short_intro = " ".join(sentences[:2])
    try:
        input_tokens = tokenizer(short_intro, return_tensors="pt", truncation=False)
        input_len = input_tokens["input_ids"].shape[1]
        max_len = min(20, max(5, int(input_len * 0.5)))
        summary = summarizer(short_intro, max_length=max_len, min_length=4, do_sample=False)[0]["summary_text"]
        return clean_truncated_heading(summary.strip().replace(".", ""))
    except Exception:
        return sentences[0] if sentences else "Untitled"

# === Summarize an Article ===
def summarize_article(article_text, max_points=5):
    article_text = re.sub(r'([a-z])([A-Z])', r'\1 \2', article_text)
    typo_fixes = {
        "ash oods": "flash floods",
        "itsworkers": "its workers",
        "TelegraphAct": "Telegraph Act",
        "SigachiIndustries": "Sigachi Industries",
        "rollvision": "roll revision",
        "ofGhana": "of Ghana"
    }
    for wrong, correct in typo_fixes.items():
        article_text = article_text.replace(wrong, correct)
    sentences = spacy_sent_tokenize(article_text)
    chunks = [" ".join(sentences[i:i+6]) for i in range(0, len(sentences), 6)]
    summary_sentences = []
    for chunk in chunks:
        try:
            input_tokens = tokenizer(chunk, return_tensors="pt", truncation=False)
            input_length = input_tokens["input_ids"].shape[1]
        except Exception:
            continue
        if input_length < 40:
            continue
        max_len = min(100, max(30, int(input_length * 0.5)))
        try:
            summary = summarizer(chunk, max_length=max_len, min_length=20, do_sample=False)[0]["summary_text"]
            points = spacy_sent_tokenize(summary)
            summary_sentences.extend([p.strip() for p in points if len(p.strip()) > 10])
        except Exception:
            continue
    return summary_sentences[:max_points]

# === Process File ===
def process_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    articles = split_into_articles(text)
    print(f"🧩 Detected {len(articles)} article(s) for summarization.\n")
    results = []
    for idx, article in enumerate(articles):
        heading = generate_heading(article)
        points = summarize_article(article)
        hashtags = generate_hashtags(article)
        if points:
            results.append({
                "title": heading,
                "summary_points": points,
                "hashtags": hashtags
            })
    return results

# === Save Summaries ===
def save_to_mongodb(summaries, input_path=None, mongo_uri="mongodb://localhost:27017", db_name="news_summarizer", collection_name="summaries"):


    from pymongo import MongoClient
    import hashlib
    from datetime import datetime

    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
        client.server_info()
    except Exception as e:
        print("❌ MongoDB connection failed:", e)
        return

    db = client[db_name]
    collection = db[collection_name]
    for entry in summaries:
        unique_str = entry["title"] + " ".join(entry["summary_points"])
        doc_id = hashlib.md5(unique_str.encode("utf-8")).hexdigest()

        doc = {
            "_id": doc_id,
            "heading": entry["title"],
            "summary_points": entry["summary_points"],
            "hashtags": entry["hashtags"],
            "source_file": input_path,
            "timestamp": datetime.utcnow()
        }

        collection.update_one({"_id": doc_id}, {"$set": doc}, upsert=True)
    print(f"✅ {len(summaries)} summaries saved to MongoDB.")

# === Run ===
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        print("❌ Please provide an input file path.")
        sys.exit(1)

    summary_data = process_file(input_path)
    save_to_mongodb(summary_data, input_path=input_path)
    print("✅ Summaries saved to MongoDB.")

