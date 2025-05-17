import os
import re
import sys
import json
import fasttext
import nltk
from nltk.tokenize import word_tokenize
from warcio.archiveiterator import ArchiveIterator
from trafilatura import extract
from bs4 import BeautifulSoup
from datasketch import MinHash, MinHashLSH
from datatrove.pipeline.filters import (
    GopherQualityFilter,
    FineWebQualityFilter,
    C4QualityFilter,
    GopherRepetitionFilter,
)
import requests
from emot.emo_unicode import UNICODE_EMO

# Ensure NLTK Data is correctly loaded
nltk.download('punkt')

# Ensure FastText Model is correctly loaded
FASTTEXT_MODEL_PATH = "lid.176.bin"
language_model = fasttext.load_model(FASTTEXT_MODEL_PATH)

# Ensure script is run with a WARC file argument (Needed for Slurm)
if len(sys.argv) < 2:
    print("Error: No WARC file provided. Usage: python commoncrawl_script.py <warc_file>")
    sys.exit(1)

warc_file_path = sys.argv[1]  # Get WARC file from command-line argument

# Load UT1 Blocklist
UT1_BLOCKLIST_URL = "http://dsi.ut-capitole.fr/blacklists/download/blacklists.tar.gz"
ut1_text_keywords = []
ut1_link_keywords = []

def load_ut1_blocklist():
    """Download and parse UT1 blocklist."""
    global ut1_text_keywords, ut1_link_keywords
    try:
        response = requests.get(UT1_BLOCKLIST_URL)
    except Exception as e:
        print(f"Error loading UT1 blocklist: {e}")
        ut1_text_keywords = []
        ut1_link_keywords = []

# Class Definition
class Document:
    def __init__(self, text):
        self.text = text

# Initialize DataTrove Filters
gopher_filter = GopherQualityFilter()
fineweb_filter = FineWebQualityFilter()
c4_filter = C4QualityFilter()
repetition_filter = GopherRepetitionFilter()

# Functions
def is_blocklisted(content, source_url):
    """Check if content or URL contains blocklisted words or links."""
    if any(keyword in content for keyword in ut1_text_keywords):
        return True
    if source_url and any(link in source_url for link in ut1_link_keywords):
        return True
    return False

def clean_html(content):
    """Remove HTML tags from content."""
    if "<html" in content.lower() or "<body" in content.lower():
        soup = BeautifulSoup(content, "html.parser")
        return soup.get_text(separator=" ").strip()
    return content.strip()

def detect_language(text):
    """Detect the language of the text using FastText."""
    cleaned_text = text.replace("\n", " ").strip()
    prediction = language_model.predict(cleaned_text[:1000])
    language = prediction[0][0].replace("__label__", "")
    confidence = prediction[1][0]
    return language, confidence

def remove_non_arabic_text(text):
    """Remove non-Arabic text using FastText language detection."""
    sentences = text.split("\n")
    arabic_sentences = [sentence for sentence in sentences if detect_language(sentence)[0] == "ar"]
    return "\n".join(arabic_sentences)

def has_excessive_newlines(text, threshold=0.5):
    """Check if the text has excessive newlines compared to its word count."""
    newline_count = text.count("\n")
    word_count = len(text.split())
    return newline_count > word_count * threshold

def convert_emojis(text):
    """Replace emojis with descriptive text."""
    for emot in UNICODE_EMO:
        text = text.replace(
            emot, " ".join(UNICODE_EMO[emot].replace(",", " ").replace(":", " ").split())
        )
    return text

def normalize_text(text):
    """Normalize Arabic text by removing diacritics and cleaning up."""
    text = re.sub(r"[ًٌٍَُِّْ]", "", text)  # Remove Arabic diacritics
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    return text.strip()

def deduplicate_documents(data, threshold=0.8):
    """Remove duplicates at the document level using MinHash."""
    lsh = MinHashLSH(threshold=threshold)
    unique_data = []
    for idx, record in enumerate(data):
        text = record['text']
        tokens = word_tokenize(text)
        m = MinHash()
        for token in tokens:
            m.update(token.encode("utf-8"))
        if not any(lsh.query(m)):
            lsh.insert(str(idx), m)
            unique_data.append(record)
    return unique_data

def deduplicate_sentences(text):
    """Remove duplicate sentences within a single document."""
    sentences = text.split("\n")
    unique_sentences = list(dict.fromkeys(sentences))  # Preserve order
    return "\n".join(unique_sentences)

def is_high_quality_text(text):
    """Filter out low-quality text (e.g., too short or mostly whitespace)."""
    if len(text.split()) < 4 or text.strip().count("\n") > len(text.split()) * 0.5:
        return False
    return True

def process_pipeline(warc_file, output_folder="Output", max_records=1000):
    """Main pipeline function to process a single WARC file."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    processed_data = []
    total_records = 0

    with open(warc_file, "rb") as stream:
        for record in ArchiveIterator(stream):
            if record.rec_type == "response":
                content = record.content_stream().read().decode("utf-8", errors="ignore")
                source_url = record.rec_headers.get_header("WARC-Target-URI")
                date = record.rec_headers.get_header("WARC-Date")

                # Extract meaningful text with Trafilatura
                extracted_text = extract(content)
                if not extracted_text:
                    continue

                # Clean HTML content
                cleaned_text = clean_html(extracted_text)

                # Detect primary language
                language, confidence = detect_language(cleaned_text)
                if language != "ar" or confidence < 0.95:
                    continue

                # Remove non-Arabic text
                arabic_only_text = remove_non_arabic_text(cleaned_text)

                # Check for excessive newlines
                if has_excessive_newlines(arabic_only_text):
                    print(f"Skipping source due to excessive newlines: {source_url}")
                    continue

                # Blocklist filtering
                if is_blocklisted(arabic_only_text, source_url):
                    continue

                # Normalize, remove emojis, and deduplicate sentences
                normalized_text = normalize_text(convert_emojis(arabic_only_text))
                deduplicated_text = deduplicate_sentences(normalized_text)

                # Wrap text in a mock Document object
                document = Document(text=deduplicated_text)

                # Apply quality filters
                if not gopher_filter.filter(document):
                    continue
                if not fineweb_filter.filter(document):
                    continue
                if not c4_filter.filter(document):
                    continue

                # Add metadata
                metadata = {
                    "date": date,
                    "labels": {
                        "language": language,
                        "language_score": confidence,
                    },
                    "source": source_url,
                    "token_count": len(deduplicated_text.split()),
                }

                processed_data.append({"text": deduplicated_text, "metadata": metadata})
                total_records += 1

                if total_records >= max_records:
                    print(f"Reached maximum records limit ({max_records}). Stopping.")
                    break

    # Deduplicate across documents
    processed_data = deduplicate_documents(processed_data)

    # Save processed data
    output_file_path = os.path.join(output_folder, f"processed_{os.path.basename(warc_file)}.json")
    with open(output_file_path, "w", encoding="utf-8") as json_file:
        json.dump(processed_data, json_file, ensure_ascii=False, indent=4)

    print(f"Processed {total_records} records. Output saved to {output_file_path}")

process_pipeline(warc_file_path)
