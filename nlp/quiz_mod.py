# quiz_module.py

import random
import re


# =====================================================
# Utility: Word Match (case + singular/plural safe)
# =====================================================
def word_present(word, sentence):
    pattern = r'\b' + re.escape(word) + r's?\b'
    return re.search(pattern, sentence, re.IGNORECASE)


# =====================================================
# Fill in the Blank
# =====================================================
def generate_fib(top_keywords, summary_sentences, num_options=4):

    fib_questions = []
    keywords = [word for word, score in top_keywords]
    used = set()

    for sentence in summary_sentences:
        for word in keywords:

            if word_present(word, sentence) and word not in used:

                pattern = r'\b' + re.escape(word) + r's?\b'
                question_text = re.sub(pattern, "____", sentence, count=1, flags=re.IGNORECASE)

                wrong_pool = [k for k in keywords if k != word]
                wrong_options = random.sample(wrong_pool, min(num_options-1, len(wrong_pool)))

                options = wrong_options + [word]
                random.shuffle(options)

                fib_questions.append({
                    "type": "FIB",
                    "question_text": question_text,
                    "options": options,
                    "answer": word
                })

                used.add(word)
                break

    return fib_questions


# =====================================================
# True / False
# =====================================================
def generate_tf(summary_sentences, top_keywords):

    tf_questions = []
    keywords = [word for word, score in top_keywords]

    for sentence in summary_sentences:

        is_true = random.choice([True, False])
        statement = sentence

        if not is_true:
            for word in keywords:
                if word_present(word, sentence):
                    replacement_pool = [k for k in keywords if k != word]
                    if replacement_pool:
                        replacement = random.choice(replacement_pool)
                        pattern = r'\b' + re.escape(word) + r's?\b'
                        statement = re.sub(pattern, replacement, sentence, count=1, flags=re.IGNORECASE)
                    break

        tf_questions.append({
            "type": "TF",
            "question_text": statement,
            "answer": is_true
        })

    return tf_questions


# =====================================================
# MCQ (Single Correct)
# =====================================================
def generate_mcq(summary_sentences, top_keywords, num_options=4):

    mcq_questions = []
    keywords = [word for word, score in top_keywords]

    for sentence in summary_sentences:
        for correct in keywords:

            if word_present(correct, sentence):

                pattern = r'\b' + re.escape(correct) + r's?\b'
                question_text = re.sub(pattern, "____", sentence, count=1, flags=re.IGNORECASE)

                wrong_pool = [k for k in keywords if k != correct]
                wrong_options = random.sample(wrong_pool, min(num_options-1, len(wrong_pool)))

                options = wrong_options + [correct]
                random.shuffle(options)

                mcq_questions.append({
                    "type": "MCQ",
                    "question_text": question_text,
                    "options": options,
                    "answer": correct
                })

                break

    return mcq_questions


# =====================================================
# MSQ (Multiple Correct)
# =====================================================
def generate_msq(summary_sentences, top_keywords):

    msq_questions = []
    keywords = [word for word, score in top_keywords]

    for sentence in summary_sentences:

        present = [k for k in keywords if word_present(k, sentence)]

        if len(present) >= 2:

            correct_answers = random.sample(present, 2)
            wrong_pool = [k for k in keywords if k not in correct_answers]

            wrong_options = random.sample(wrong_pool, min(2, len(wrong_pool)))

            options = correct_answers + wrong_options
            random.shuffle(options)

            msq_questions.append({
                "type": "MSQ",
                "question_text": f"Select all correct options related to:\n{sentence}",
                "options": options,
                "answer": correct_answers
            })

    return msq_questions