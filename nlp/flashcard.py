try:
    from nlp.nlp_pipeline import get_summary_and_keywords
except ImportError:
    from nlp_pipeline import get_summary_and_keywords
import spacy
import json
import os
import webbrowser

nlp = spacy.load("en_core_web_sm")


# ---------------- EXTRACT TRUE TECHNICAL TERMS ----------------
def extract_technical_terms(summary_sentences):
    terms = set()

    for sentence in summary_sentences:
        doc = nlp(sentence)

        for chunk in doc.noun_chunks:
            text = chunk.text.strip()

            # Keep only 1–3 word phrases
            if 1 <= len(text.split()) <= 3:

                # Remove chunks that are mostly stopwords
                if all(token.is_stop for token in chunk):
                    continue

                # Must contain at least one noun or proper noun
                if any(token.pos_ in ["NOUN", "PROPN"] for token in chunk):

                    # Remove phrases that start with pronouns/determiners
                    if chunk[0].pos_ in ["PRON", "DET"]:
                        continue

                    # Avoid very small words
                    if len(text) > 4:
                        terms.add(text)

    return list(terms)


# ---------------- FLASHCARD GENERATOR ----------------
def generate_flashcards(summary_sentences):

    flashcards = []
    technical_terms = extract_technical_terms(summary_sentences)

    # 1. For each term, find its best (shortest) explanation sentence
    term_to_explanation = {}
    for term in technical_terms:
        candidate_sentences = [s.strip() for s in summary_sentences if term.lower() in s.lower()]
        if not candidate_sentences:
            continue

        # The shortest sentence is likely the most focused definition
        best_explanation = min(candidate_sentences, key=len)
        if best_explanation:
            term_to_explanation[term] = best_explanation

    # 2. Group terms by their explanation to avoid duplicate card backs
    explanation_to_terms = {}
    for term, explanation in term_to_explanation.items():
        if explanation not in explanation_to_terms:
            explanation_to_terms[explanation] = []
        explanation_to_terms[explanation].append(term)

    # 3. Create one flashcard per unique explanation
    for explanation, terms in explanation_to_terms.items():
        # For the card's front, use the longest (most specific) term from the group
        main_term = max(terms, key=len)

        # Ensure the explanation sentence has proper punctuation
        final_explanation = explanation
        if not final_explanation.endswith((".", "!", "?")):
            final_explanation += "."

        flashcards.append({
            "front": main_term.title(),
            "back": final_explanation
        })

    return flashcards


# ---------------- FLASHCARD VIEWER ----------------
def run_flashcards(flashcards):
    # Determine paths relative to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    front_dir = os.path.join(base_dir, "..", "front")
    data_file = os.path.join(front_dir, "flashcards_data.js")
    html_file = os.path.join(front_dir, "flashcards.html")

    # Save flashcards as a JS variable
    js_content = f"const generatedFlashcards = {json.dumps(flashcards, indent=4)};"

    try:
        with open(data_file, "w", encoding="utf-8") as f:
            f.write(js_content)
        print(f"\n✅ Flashcards generated and saved to {data_file}")
        print("🚀 Opening Flashcards in browser...")
        webbrowser.open(html_file)
    except Exception as e:
        print(f"❌ Error saving flashcards: {e}")


# ---------------- MAIN ----------------
if __name__ == "__main__":

    file_path = input("Enter file path (PDF or DOCX): ")

    summary_sentences, keywords = get_summary_and_keywords(file_path)

    flashcards = generate_flashcards(summary_sentences)

    run_flashcards(flashcards)