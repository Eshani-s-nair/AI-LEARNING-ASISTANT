from unittest import result

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from nlp.nlp_pipeline import get_summary_and_keywords, improve_summary_with_ollama, show_summary_in_browser, generate_title_with_ollama, read_pdf, read_docx
from nlp.nlp_pipeline2 import analyze_question_papers, nlp
from nlp.quiz_mod import generate_fib, generate_tf, generate_mcq, generate_msq
from nlp.flashcard import generate_flashcards
from nlp.db import DatabaseManager
import os
import requests
import json
import logging
import re
import traceback
import shutil
import warnings

# Suppress the Torch UserWarning about pin_memory and accelerators
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
logging.getLogger("torch.utils.data.dataloader").setLevel(logging.ERROR)
# --- EASYOCR CONFIGURATION ---
import easyocr
try:
    print("🚀 Initializing EasyOCR Reader (English)...")
    # This will automatically download models on the first run
    reader = easyocr.Reader(['en'])
    print("✅ EasyOCR initialized successfully.")
except Exception as e:
    print(f"❌ Failed to initialize EasyOCR: {e}")
    reader = None

# --- POPPLER CONFIGURATION ---
# Poppler is needed by pdf2image (used for OCRing PDFs)
POPPLER_POSSIBLE_PATHS = [
    r'C:\Program Files\poppler\Library\bin',
    r'C:\poppler\Library\bin'
]
POPPLER_FOUND = False
for path in POPPLER_POSSIBLE_PATHS:
    if os.path.exists(path):
        if path not in os.environ['PATH']:
            os.environ['PATH'] = path + os.pathsep + os.environ['PATH']
        POPPLER_FOUND = True
        print(f"✅ Poppler detected at: {path}")
        break

# Suppress PyPDF2 metadata warnings
logging.getLogger("PyPDF2").setLevel(logging.ERROR)

# You need to install Flask first:
# pip install Flask

# Use absolute path to ensure Flask finds the 'front' folder correctly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONT_DIR = os.path.join(BASE_DIR, 'front')

app = Flask(__name__, static_folder=FRONT_DIR, static_url_path='')
CORS(app) # Enable Cross-Origin Resource Sharing

@app.route('/')
def home():
    # Serve home.html as the main entry point
    if os.path.exists(os.path.join(app.static_folder, 'home.html')):
        return send_from_directory(app.static_folder, 'home.html')
    else:
        # Clear error message if the file is missing
        return "<h1>Error</h1><p>No 'home.html' found in the 'front' folder.</p>", 404

@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    first_name = data.get('firstName')
    last_name = data.get('lastName')
    email = data.get('email')
    username = data.get('username')
    password = data.get('password')

    if not all([first_name, last_name, email, username, password]):
        return jsonify({"error": "Missing required fields"}), 400

    db = DatabaseManager()
    result = db.create_user(first_name, last_name, email, username, password)

    if result.get("success"):
        return jsonify(result), 201
    else:
        status_code = 409 if "exists" in result.get("message", "") else 500
        return jsonify({"error": result.get("message", "An unknown error occurred.")}), status_code

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    username_or_email = data.get('username')
    password = data.get('password')

    if not all([username_or_email, password]):
        return jsonify({"error": "Missing username or password"}), 400

    db = DatabaseManager()
    result = db.authenticate_user(username_or_email, password)

    if result.get("success"):
        return jsonify(result), 200
    else:
        return jsonify({"error": result.get("message", "Authentication failed")}), 401

@app.route('/process-document', methods=['POST'])
def process_document():
    user_id_raw = request.form.get('user_id')
    if not user_id_raw:
        return jsonify({"error": "Authentication required"}), 401
    user_id = int(user_id_raw)

    pasted_text = request.form.get('pasted_text', '').strip()
    file = request.files.get('file')

    if not file and not pasted_text:
        return jsonify({"error": "No file or text provided"}), 400

    try:
        if file and file.filename != '':
            # Save file temporarily
            upload_folder = os.path.join(os.getcwd(), 'uploads')
            os.makedirs(upload_folder, exist_ok=True)
            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)
            summary_sentences, keywords = get_summary_and_keywords(file_path=file_path)
            file_path_for_db = file_path
        else:
            # Process pasted text directly
            summary_sentences, keywords = get_summary_and_keywords(raw_text=pasted_text)
            file_path_for_db = "Pasted Text"

        if not summary_sentences:
            return jsonify({"error": "Failed to process content. Text might be too short."}), 400
            
        # 2. Refine summary with AI (Ollama)
        refined_summary = ""
        try:
            # This is a non-blocking call; if Ollama isn't running, it will just be empty.
            refined_summary = improve_summary_with_ollama(summary_sentences)
        except Exception as e:
            print(f"⚠️  Warning: Ollama refinement failed. {e}")
            
        # 2.5 Generate Title
        title = generate_title_with_ollama(summary_sentences)

        # 3. Generate other study materials
        flashcards = generate_flashcards(summary_sentences)
        fib_questions = generate_fib(keywords, summary_sentences)
        tf_questions = generate_tf(summary_sentences, keywords)
        mcq_questions = generate_mcq(summary_sentences, keywords)
        msq_questions = generate_msq(summary_sentences, keywords)

        # --- Save Flashcards & Quiz Data to Frontend Files ---
        front_dir = app.static_folder
        # os.makedirs(front_dir, exist_ok=True) # Static folder usually exists

        with open(os.path.join(front_dir, "flashcards_data.js"), "w", encoding="utf-8") as f:
            f.write(f"const generatedFlashcards = {json.dumps(flashcards, indent=4)};")

        summary_data_output = {
            "summary": summary_sentences,
            "keywords": [k[0] for k in keywords],
            "refined_summary": refined_summary
        }
        with open(os.path.join(front_dir, "summary_data.js"), "w", encoding="utf-8") as f:
            f.write(f"const generatedSummaryData = {json.dumps(summary_data_output, indent=4)};")

        quiz_data_output = {
            "fill_in_the_blank": fib_questions,
            "true_false": tf_questions,
            "mcq": mcq_questions,
            "msq": msq_questions
        }
        with open(os.path.join(front_dir, "quiz_data.js"), "w", encoding="utf-8") as f:
            f.write(f"const generatedQuizData = {json.dumps(quiz_data_output, indent=4)};")

        # 4. Save original summary to database for the authenticated user
        db = DatabaseManager()
        db.save_data(file_path_for_db, summary_sentences, keywords, user_id=user_id, title=title)

        # Generate the summary_data.js file and open the browser preview
        # show_summary_in_browser(summary_sentences, keywords, refined_summary)

        # 5. Return everything as a single JSON response
        return jsonify({
            # "file_path": file_path, # Not needed by the frontend
            "summary": summary_sentences,
            "refined_summary": refined_summary,
            "keywords": [k[0] for k in keywords],
            "flashcards": flashcards,
            "quiz": {
                "fill_in_the_blank": fib_questions,
                "true_false": tf_questions,
                "mcq": mcq_questions,
                "msq": msq_questions
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def refine_question_data(topic_data):
    """
    Deduplicates 'almost same' questions and enforces the 'one topic per question' rule.
    Cleans questions by removing marks, question numbers, and OCR noise for better readability.
    """
    if not topic_data or not isinstance(topic_data, list):
        return topic_data

    unique_questions = {} # text -> question_obj
    question_to_topic = {} # text -> single topic name

    for entry in topic_data:
        # 1. Clean Topic Name (e.g. "Module 1: BFS" -> "bfs")
        raw_topic = entry.get('topic', 'general')
        topic_name = re.sub(r'(?i)^(?:module|unit|chapter|part|section|no)\s*\d+[:\- ]*', '', raw_topic).strip().lower()
        topic_name = re.sub(r'\s+', ' ', topic_name) or 'general'
        
        if topic_name.isdigit() or len(topic_name) < 2:
            topic_name = 'general'
        
        for q in entry['questions']:
            raw_text = q['text']
            
            # 2. READABILITY CLEANING
            cleaned = re.sub(r'[\n\r\t]+', ' ', raw_text)
            
            # Strip leading question numbering (e.g., "1.", "Q1.", "(a)", "iv.")
            cleaned = re.sub(r'(?i)^(?:Q|Question|No|Part|Section|Unit|Module)?\s*\d*[a-zivx]{0,3}[\s\.)\-:]+', '', cleaned)
            
            # Remove Bloom's/CO levels (e.g., "L3", "(CO2)", "[L1]")
            cleaned = re.sub(r'(?i)\s*[\(\[]?\b(?:L|CO)[1-6]\b[\)\]]?\s*', ' ', cleaned)
            
            # Remove trailing marks (e.g., "(10 marks)", "[5]", " (5+5) ")
            cleaned = re.sub(r'(?i)\s*[\(\[]\s*\d+\s*(?:\+\s*\d+\s*)*marks?\s*[\)\]]\s*$', '', cleaned)
            cleaned = re.sub(r'\s*[\(\[]\s*\d+\s*[\)\]]\s*$', '', cleaned)
            
            # Remove common exam metadata noise
            cleaned = re.sub(r'(?i)PART [A-D].*', '', cleaned)
            
            # Remove clusters of noise characters (OCR junk)
            cleaned = re.sub(r"[^\w\s.,?()\-:;'\"+*/={}]{2,}|[?@#$]{1,}", ' ', cleaned)
            
            # Final trim and whitespace collapse
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            
            # 3. JUNK FILTERING
            alnum_count = sum(1 for char in cleaned if char.isalnum())
            density = alnum_count / len(cleaned) if cleaned else 0
            if len(cleaned) < 15 or density < 0.6:
                continue
            
            q_key = cleaned.lower()

            # 4. SIMILARITY GROUPING (One Category per Question)
            found_match = False
            for existing_key in unique_questions.keys():
                set1, set2 = set(q_key.split()), set(existing_key.split())
                # Jaccard similarity threshold (0.7 means 70% overlap)
                similarity = len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0
                
                if similarity > 0.70:
                    found_match = True
                    count = q.get('occurrence_count', 1)
                    unique_questions[existing_key]['occurrence_count'] = unique_questions[existing_key].get('occurrence_count', 1) + count
                    break
            
            if not found_match:
                q['text'] = cleaned
                if 'occurrence_count' not in q:
                    q['occurrence_count'] = 1
                unique_questions[q_key] = q
                question_to_topic[q_key] = topic_name

    # 5. Reconstruct data grouping by the single assigned topic
    final_map = {}
    for q_key, topic in question_to_topic.items():
        final_map.setdefault(topic, []).append(unique_questions[q_key])

    return [{"topic": t, "questions": qs, "frequency": len(qs), 
             "importance_score": sum(q.get('occurrence_count', 1) for q in qs)} 
            for t, qs in final_map.items() if qs]

@app.route('/process-question-papers', methods=['POST'])
def process_question_papers():
    """Processes up to 10 question papers to find high-frequency important questions."""
    user_id_raw = request.form.get('user_id')
    if not user_id_raw:
        return jsonify({"error": "Authentication required"}), 401

    files = request.files.getlist('files') if 'files' in request.files else []
    pasted_text = request.form.get('pasted_text', '').strip()

    if not files and not pasted_text:
        return jsonify({"error": "No files or text provided"}), 400

    if len(files) > 10:
        return jsonify({"error": "Maximum 10 files allowed."}), 400

    upload_folder = os.path.join(os.getcwd(), 'uploads', 'question_papers')
    os.makedirs(upload_folder, exist_ok=True)
    
    saved_paths = []
    for file in files:
        if file.filename.lower().endswith('.pdf'):
            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)
            saved_paths.append(file_path)

    if pasted_text:
        pasted_file_path = os.path.join(upload_folder, f"pasted_questions_{user_id_raw}.txt")
        with open(pasted_file_path, "w", encoding="utf-8") as f:
            f.write(pasted_text)
        saved_paths.append(pasted_file_path)

    if not saved_paths:
        return jsonify({"error": "No valid PDF files found."}), 400

    try:
        # Run the new NLP pipeline for question analysis
        report_path = os.path.join(upload_folder, f"analysis_report_{user_id_raw}.json")
        
        # Pre-check OCR dependencies
        try:
            import easyocr
        except ImportError:
            return jsonify({"success": False, "error": "easyocr library is missing. Run 'pip install easyocr'."}), 500

        if not POPPLER_FOUND:
            return jsonify({
                "success": False, 
                "error": "Poppler not found. OCR requires Poppler to be installed at C:\\Program Files\\poppler\\Library\\bin"
            }), 500

        try:
            # Ensure we are passing valid paths
            if not saved_paths:
                return jsonify({"success": False, "error": "No files were saved correctly."}), 400

            print(f"📊 Starting analysis of {len(saved_paths)} question papers...")
            result = analyze_question_papers(saved_paths, output_report_path=report_path)
            if not result or not isinstance(result, list):
                return jsonify({"success": False, "error": "No questions could be extracted from the provided files."}), 400

            important_questions = result
            print("✅ Question extraction complete.")
            
            if isinstance(result, dict) and "error" in result:
                 return jsonify({
                    "success": False, 
                    "error": f"Analysis Error: {result['error']}"
                }), 500
            print(f"🔍 Refining {len(result) if isinstance(result, list) else 'extracted'} topics/questions...")

            important_questions = result# Call the refinement function here
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Analysis Error Detail: {error_msg}")
            traceback.print_exc() # This will show the real error in your terminal
            return jsonify({"success": False, "error": error_msg}), 500

        print("✨ Question data refined and deduplicated.")
        
        total_questions = 0
        if isinstance(important_questions, list) and len(important_questions) > 0:
            for topic in important_questions:
                if isinstance(topic, dict) and 'questions' in topic:
                    total_questions += len(topic.get('questions', []))
        elif isinstance(important_questions, dict) and "questions" in important_questions:
            total_questions = len(important_questions["questions"])

        # If the result is an error dictionary or empty
        if isinstance(important_questions, dict) and "error" in important_questions:
            return jsonify({"success": False, "error": important_questions["error"]}), 400
            
        if not important_questions or total_questions == 0:
            return jsonify({"success": False, "error": "Analysis failed: No questions extracted. If these are scanned PDFs, ensure you are running the server as Administrator so OCR can function."}), 400

        # Save to frontend JS file for display in refinedques.html
        front_dir = app.static_folder
        js_content = f"const generatedRefinedQuestions = {json.dumps(important_questions, indent=4)};"
        with open(os.path.join(front_dir, "refined_questions_data.js"), "w", encoding="utf-8") as f:
            f.write(js_content)
        print("💾 Refined questions saved to JS file.")

        return jsonify({"success": True, "questions": important_questions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get-ai-answer', methods=['POST'])
def get_ai_answer():
    """Uses Ollama to generate an answer for a specific question."""
    data = request.get_json()
    question = data.get('question')
    topic = data.get('topic', 'General')
    user_id = data.get('user_id')

    if not question:
        return jsonify({"error": "Question is required"}), 400

    context_snippet = ""
    # Attempt to find relevant context in the user's latest textbook
    if user_id:
        try:
            db = DatabaseManager()
            history = db.get_documents_by_user(int(user_id))
            if history:
                latest_doc = history[0]
                file_path = os.path.join(os.getcwd(), 'uploads', latest_doc['file_name'])
                
                if os.path.exists(file_path):
                    text = read_pdf(file_path) if file_path.endswith('.pdf') else read_docx(file_path)
                    
                    if text:
                        # Identify core keywords from the question to find relevant sentences in textbook
                        q_keywords = [t.lemma_ for t in nlp(question.lower()) if not t.is_stop and t.is_alpha]
                        doc = nlp(text)
                        
                        relevant_sents = []
                        for sent in doc.sents:
                            score = sum(1 for kw in q_keywords if kw in sent.text.lower())
                            if score > 0:
                                relevant_sents.append((score, sent.text.strip()))
                        
                        # Sort by match score and take the top 5 sentences as context
                        relevant_sents.sort(key=lambda x: x[0], reverse=True)
                        context_snippet = " ".join([s[1] for s in relevant_sents[:5]])
        except Exception as e:
            print(f"⚠️ Warning: Context search failed. {e}")

    prompt = f"""You are an expert professor and exam coach for B.Tech Computer Science students.

            Topic: {topic}
            Question: {question}
            {f"Relevant textbook context: {context_snippet}" if context_snippet else ""}

            Instructions:
            - Give a precise, exam-ready answer a B.Tech student needs to score full marks
            - Start with a 1-line direct definition or answer
            - Then explain with technical depth using bullet points
            - Include an example, diagram description, or pseudocode if relevant
            - End with a "Key Points to Remember" section with 3-5 bullet points
            - Do NOT include unnecessary filler phrases like "Great question!" or "I hope this helps"
            - Be technically accurate and concise

            Answer:"""
                
    try:
        # Using the same Ollama setup as nlp_pipeline.py
        print(f"🤖 Requesting AI answer for: {question[:50]}...")
        url = "http://localhost:11434/api/generate"
        response = requests.post(url, json={
            "model": "phi3:mini",
            "prompt": prompt,
            "stream": False,
            "temperature": 0.4
        }, timeout=60)
        if response.status_code == 200:
            return jsonify({"answer": response.json().get("response")})
        else:
            return jsonify({"error": "Ollama returned an error status"}), response.status_code
    except Exception as e:
        return jsonify({"error": f"AI service unavailable: {str(e)}"}), 500

@app.route('/user/history', methods=['GET'])
def get_user_history():
    """Returns a list of documents uploaded by a user."""
    user_id_raw = request.args.get('user_id')
    if not user_id_raw:
        return jsonify({"error": "User ID is required"}), 400
    user_id = int(user_id_raw)

    db = DatabaseManager()
    documents = db.get_documents_by_user(user_id)
    return jsonify(documents)

@app.route('/load-history/<int:doc_id>', methods=['POST'])
def load_history(doc_id):
    """
    Regenerates the JS files for a past document so the frontend can display them.
    """
    try:
        db = DatabaseManager()
        data = db.get_summary_by_id(doc_id)
        
        if not data:
            return jsonify({"error": "Document not found"}), 404
            
        # Reconstruct data
        summary_text = data['summary']
        # Split text back into sentences (simple approximation by period)
        summary_sentences = [s.strip() for s in summary_text.split('.') if s.strip()]
        keywords = [(k, 1.0) for k in data['keywords']] # Mock score as 1.0

        # Regenerate Quizzes and Flashcards
        flashcards = generate_flashcards(summary_sentences)
        fib = generate_fib(keywords, summary_sentences)
        tf = generate_tf(summary_sentences, keywords)
        mcq = generate_mcq(summary_sentences, keywords)
        msq = generate_msq(summary_sentences, keywords)

        # Save to frontend files (Overwriting current session)
        front_dir = app.static_folder
        
        with open(os.path.join(front_dir, "flashcards_data.js"), "w", encoding="utf-8") as f:
            f.write(f"const generatedFlashcards = {json.dumps(flashcards, indent=4)};")

        quiz_data_output = {"fill_in_the_blank": fib, "true_false": tf, "mcq": mcq, "msq": msq}
        with open(os.path.join(front_dir, "quiz_data.js"), "w", encoding="utf-8") as f:
            f.write(f"const generatedQuizData = {json.dumps(quiz_data_output, indent=4)};")

        # Note: Refined summary is not stored in DB currently, so passing empty string
        show_summary_in_browser(summary_sentences, keywords, "")

        return jsonify({"success": True, "message": "History loaded"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/summary/<int:doc_id>', methods=['GET'])
def get_summary(doc_id):
    db = DatabaseManager()
    data = db.get_summary_by_id(doc_id)
    if data:
        return jsonify(data)
    else:
        return jsonify({"error": "Summary not found"}), 404


if __name__ == '__main__':
    print("Starting Flask server...")
    print("Server accessible at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)