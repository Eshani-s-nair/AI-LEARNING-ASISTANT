# -------------------- IMPORTS --------------------
import spacy
from PyPDF2 import PdfReader
from docx import Document
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
try:
    from nlp.db import DatabaseManager
except ImportError:
    from db import DatabaseManager
import requests
import os
import json
import webbrowser


# -------------------- FILE READERS --------------------
def read_pdf(file_path):
    reader = PdfReader(file_path)
    return " ".join(page.extract_text() for page in reader.pages if page.extract_text())


def read_docx(file_path):
    doc = Document(file_path)
    return "\n".join(p.text for p in doc.paragraphs)


# -------------------- NLP SUMMARY + KEYWORDS --------------------
def get_summary_and_keywords(file_path=None, top_k=10, summary_k=3, raw_text=None):
    nlp = spacy.load("en_core_web_sm")
    stop_words = set(stopwords.words("english"))

    if raw_text:
        text = raw_text
    else:
        # Read file
        if file_path.endswith(".pdf"):
            text = read_pdf(file_path)
        elif file_path.endswith(".docx"):
            text = read_docx(file_path)
        else:
            raise ValueError("Unsupported file format")

    doc = nlp(text)

    clean_sentences = []
    original_sentences = []

    for sent in doc.sents:
        original_sentences.append(sent.text.strip())
        tokens = [
            token.lemma_.lower()
            for token in sent
            if token.is_alpha and token.text.lower() not in stop_words
        ]
        clean_sentences.append(" ".join(tokens))

    # TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(clean_sentences)
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray().sum(axis=0)

    # Top Keywords
    keywords = sorted(
        zip(feature_names, scores),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    # Sentence Scoring
    sentence_scores = {}
    for sent, cleaned in zip(original_sentences, clean_sentences):
        score = sum(
            scores[list(feature_names).index(w)]
            for w in cleaned.split()
            if w in feature_names
        )
        sentence_scores[sent] = score

    summary = sorted(
        sentence_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:summary_k]

    summary_sentences = [s for s, _ in summary]

    return summary_sentences, keywords


# -------------------- OLLAMA REFINEMENT --------------------
def improve_summary_with_ollama(summary_sentences):
    url = "http://localhost:11434/api/generate"

    summary_text = " ".join(summary_sentences)

    prompt = f"""
You are an expert academic assistant specializing in B.Tech subjects.

Rewrite the following summary into high-quality, professional study notes for a student.

Formatting Rules:
1. Use Clear Headings: Break the content into logical sections if possible.
2. Bold Key Terms: Use **Keyword** or **Term:** for definitions and important technical terms.
3. Formulas: Present mathematical formulas or equations on their own lines, prefixed with "Formula: ".
4. Bullet Points: Use structured lists for features, advantages, or steps.
5. Quality: Simplify complex sentences while maintaining technical accuracy.
6. Fidelity: Use ONLY the given content. Do NOT introduce new outside information.

Text:
{summary_text}

Now generate the improved notes.
"""

    response = requests.post(
        url,
        json={
            "model": "phi3:mini",
            "prompt": prompt,
            "stream": False,
            "temperature": 0.3
        }
    )

    data = response.json()
    return data["response"]

def generate_title_with_ollama(summary_sentences):
    """Generates a 3-5 word title for the study session."""
    url = "http://localhost:11434/api/generate"
    summary_text = " ".join(summary_sentences[:2])
    prompt = f"Based on this summary, generate a short 3 to 5 word title for a study session. Output ONLY the title, no quotes: {summary_text}"
    
    try:
        response = requests.post(
            url,
            json={
                "model": "phi3:mini",
                "prompt": prompt,
                "stream": False,
                "temperature": 0.5
            },
            timeout=5
        )
        return response.json().get("response", "Untitled Session").strip().replace('"', '')
    except Exception:
        return None

# -------------------- BROWSER DISPLAY --------------------
def show_summary_in_browser(summary_sentences, keywords, refined_summary):
    """
    Saves summary data to a JS file and opens the summary HTML page.
    """
    summary_data = {
        "summary": summary_sentences,
        "keywords": [kw[0] for kw in keywords],
        "refined_summary": refined_summary
    }

    base_dir = os.path.dirname(os.path.abspath(__file__))
    front_dir = os.path.join(base_dir, "..", "front")
    html_file = os.path.join(front_dir, "summary.html")
    preview_file = os.path.join(front_dir, "summary_preview.html")
    data_file = os.path.join(front_dir, "summary_data.js")

    # Ensure the front directory exists
    if not os.path.exists(front_dir):
        os.makedirs(front_dir)

    # 1. Save data to JS file (Allows summary.html to work if opened directly)
    js_content = f"const generatedSummaryData = {json.dumps(summary_data, indent=4)};"
    try:
        with open(data_file, "w", encoding="utf-8") as f:
            f.write(js_content)
        print(f"✅ Summary data saved to: {os.path.abspath(data_file)}")
    except Exception as e:
        print(f"❌ Error saving JS data: {e}")

    # 2. Create a preview file with the data embedded directly (Avoids CORS issues)
    data_script = f"<script>const generatedSummaryData = {json.dumps(summary_data, indent=4)};</script>"

    try:
        with open(html_file, "r", encoding="utf-8") as f:
            html_content = f.read()
        
        # Inject data before the closing body tag
        new_html_content = html_content.replace("</body>", f"{data_script}</body>")
        
        with open(preview_file, "w", encoding="utf-8") as f:
            f.write(new_html_content)
            
        print(f"\n✅ Summary preview generated: {preview_file}")
        print("🚀 Opening Summary in browser...")
        webbrowser.open('file://' + os.path.realpath(preview_file))
    except Exception as e:
        print(f"❌ Error showing summary in browser: {e}")


# =====================================================main func=================================================
if __name__ == "__main__":
    file_path = input("Enter file path (PDF or DOCX): ")

    # 1. NLP PROCESSING
    print("\n--- Analyzing Document ---")
    try:
        summary_sentences, keywords = get_summary_and_keywords(file_path)
        print("✅ Analysis complete.")
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        exit()

    # 2. OLLAMA REFINEMENT
    print("\n--- Refining with AI ---")
    improved_notes = ""
    try:
        improved_notes = improve_summary_with_ollama(summary_sentences)
        print("✅ AI refinement complete.")
    except Exception:
        print("⚠️ Ollama not running. Start Ollama to get improved summary. Skipping...")

    # 3. SAVE TO DATABASE
    db = DatabaseManager()
    db.save_data(file_path, summary_sentences, keywords, user_id=1)

    # 4. DISPLAY IN BROWSER
    show_summary_in_browser(summary_sentences, keywords, improved_notes)