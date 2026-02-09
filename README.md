# AI-Powered Study Assistant

**Description:**  
The AI-Powered Study Assistant is a Python tool that extracts keywords and generates concise summaries from PDFs and Word documents using NLP techniques. It helps students study efficiently and lays the foundation for interactive features like quizzes and progress tracking.

---

## **NLP Process**

The NLP module of the project follows **5 main steps**:

| Step | Description |
|------|-------------|
| **1. File Reading** | Read PDFs (`PyPDF2`) or Word documents (`python-docx`) to extract raw text. |
| **2. Text Preprocessing** | Split text into sentences and words (tokenization), remove stopwords, and lemmatize words using `spaCy` and `NLTK`. |
| **3. TF-IDF Vectorization** | Convert cleaned sentences into numerical vectors using `scikit-learn`'s `TfidfVectorizer` to score word importance. |
| **4. Keyword Extraction** | Identify top keywords based on TF-IDF scores, representing the most important concepts. |
| **5. Extractive Summarization** | Score sentences using TF-IDF word scores and select top sentences as a concise summary. |

---

## **Libraries Used**

| Library | Purpose |
|---------|---------|
| `spaCy` | NLP processing: tokenization, lemmatization, and sentence segmentation. |
| `NLTK` | Provides English stopwords for text cleaning. |
| `PyPDF2` | Reads PDF files and extracts text. |
| `python-docx` | Reads Word (.docx) documents and extracts text. |
| `scikit-learn` | TF-IDF vectorization to score words and sentences numerically. |

---

## **Usage**

1. Place your PDF or DOCX file in the project folder (e.g., `sample_files/`).  
2. Update the `file_path` variable in `nlp_pipeline.py`.  
3. Run the script:

```bash
python nlp_pipeline.py
