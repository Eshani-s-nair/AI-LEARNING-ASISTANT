import spacy
from PyPDF2 import PdfReader
from docx import Document
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Function to read PDF files
def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to read Word documents
def read_docx(file_path):
    doc = Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text


# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load stopwords
stop_words = set(stopwords.words("english"))


file_path = "sample_files/Test_pdf.pdf"  # or notes.docx
if file_path.endswith(".pdf"):
    text = read_pdf(file_path)
elif file_path.endswith(".docx"):
    text = read_docx(file_path)
else:
    print("Unsupported file format")
    exit()


# Process text
doc = nlp(text)

print("STEP 1: Original Sentences\n")
for sent in doc.sents:
    print(sent)

print("\nSTEP 2: Cleaned Sentences\n")

clean_sentences = []

for sent in doc.sents:
    tokens = []
    for token in sent:
        if token.is_alpha and token.text.lower() not in stop_words:
            tokens.append(token.lemma_.lower())

    cleaned = " ".join(tokens)
    clean_sentences.append(cleaned)
    print(cleaned)
# Step 3: TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(clean_sentences)

# Get all words
feature_names = vectorizer.get_feature_names_out()

# Sum scores for each word across sentences
scores = tfidf_matrix.toarray().sum(axis=0)
# Pair words with their scores
keywords = list(zip(feature_names, scores))

# Sort by score descending
keywords = sorted(keywords, key=lambda x: x[1], reverse=True)

# Pick top 10 keywords
top_keywords = keywords[:10]

print("\nTop Keywords:")
for word, score in top_keywords:
    print(word)

# Step 4: Sentence scoring for summarization
sentence_scores = {}

for sent, cleaned in zip(doc.sents, clean_sentences):
    score = 0
    for word in cleaned.split():
        # Find TF-IDF score for each word
        if word in feature_names:
            index = list(feature_names).index(word)
            score += scores[index]
    sentence_scores[sent.text] = score
# Pick top 2 sentences for summary
summary_sentences = sorted(
    sentence_scores.items(),
    key=lambda x: x[1],
    reverse=True
)[:2]

print("\nSummary:")
for s, _ in summary_sentences:
    print(s)

