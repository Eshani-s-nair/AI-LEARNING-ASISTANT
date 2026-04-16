import spacy
from PyPDF2 import PdfReader
from collections import Counter, defaultdict
import easyocr
import numpy as np
import re
import os
import json
import shutil

# OCR Libraries
try:
    from pdf2image import convert_from_path
    from PIL import Image

    # Initialize EasyOCR reader (this may download models on the first run)
    print("🚀 Initializing EasyOCR Reader in pipeline...")
    reader = easyocr.Reader(['en'])

    # 2. Path to Poppler's 'bin' directory.
    # PLEASE VERIFY THIS PATH in your File Explorer!
    POPPLER_POSSIBLE_PATHS = [
        r'C:\Program Files\poppler\Library\bin',
        r'C:\poppler\Library\bin',
        r'C:\Program Files (x86)\poppler\Library\bin',
        r'C:\poppler-0.68.0\bin'
    ]
    POPPLER_PATH = None
    for path in POPPLER_POSSIBLE_PATHS:
        if os.path.exists(path) and path not in os.environ['PATH']:
            POPPLER_PATH = path
            os.environ['PATH'] = path + os.pathsep + os.environ['PATH']
            break

    HAS_OCR = True
except ImportError:
    print("OCR libraries (easyocr, pdf2image) not found. Standard text extraction only.")
    HAS_OCR = False

# Ensure the spaCy model is available
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(file_path):
    """Reads a PDF and returns its full text content."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
        
    # If the text is suspiciously short, it might be a scanned PDF. Try OCR.
    if len(text.strip()) < 100:
        print(f"🔍 Scanned PDF detected: {os.path.basename(file_path)}. Running EasyOCR...")
        text = extract_text_via_ocr(file_path)
        
    return text

def extract_text_via_ocr(file_path):
    """Converts PDF pages to images and performs OCR."""
    if not HAS_OCR:
        raise ImportError("OCR libraries (pdf2image, easyocr) are not installed.")
        
    if POPPLER_PATH and not os.path.exists(POPPLER_PATH):
        raise FileNotFoundError(f"POPPLER_PATH does not exist: {POPPLER_PATH}. Ensure Poppler is installed.")
    
    # 300 DPI is standard for high-quality OCR
    images = convert_from_path(file_path, 300, poppler_path=POPPLER_PATH)
    ocr_text = ""
    for img in images:
        # EasyOCR processes numpy arrays
        img_np = np.array(img)
        results = reader.readtext(img_np, detail=0)
        ocr_text += " ".join(results) + "\n"
    return ocr_text

def identify_questions(text):
    """Extracts candidate questions or exam prompts from raw text."""
    doc = nlp(text)
    questions = []
    
    # Matches patterns like "Q1.", "1.", "Question 5:", etc.
    header_regex = re.compile(r'^(?:Q(?:uestion)?\s*\d+[:.]?|\d+[:.]?)\s*', re.IGNORECASE)

    for sent in doc.sents:
        s = sent.text.strip()
        
        # Criteria for identifying an exam question:
        # 1. Ends with a question mark.
        # 2. Starts with a question identifier (e.g., "1.").
        # 3. Starts with common directive verbs used in exams.
        is_question_like = (
            s.endswith('?') or 
            header_regex.match(s) or
            any(s.lower().startswith(word) for word in [
                "explain", "define", "describe", "discuss", "calculate", 
                "compare", "list", "identify", "what", "how", "why"
            ])
        )
        
        # Filter for quality: exclude very short fragments or noise
        if is_question_like and len(s.split()) > 4:
            cleaned_s = header_regex.sub('', s)
            questions.append(cleaned_s)
            
    return questions

# Add this at the top of nlp_pipeline2.py with other imports
import string

# Predefined technical topic taxonomy - maps keywords to clean topic names
TECH_TOPIC_MAP = {
    # Algorithms & DS
    "bfs": "Breadth First Search (BFS)",
    "dfs": "Depth First Search (DFS)",
    "dijkstra": "Dijkstra's Algorithm",
    "kruskal": "Kruskal's Algorithm",
    "prim": "Prim's Algorithm",
    "bellman": "Bellman-Ford Algorithm",
    "floyd": "Floyd-Warshall Algorithm",
    "dynamic programming": "Dynamic Programming",
    "greedy": "Greedy Algorithms",
    "backtracking": "Backtracking",
    "divide and conquer": "Divide and Conquer",
    "sorting": "Sorting Algorithms",
    "searching": "Searching Algorithms",
    "binary search": "Binary Search",
    "hashing": "Hashing",
    "heap": "Heap Data Structure",
    "tree": "Trees",
    "binary tree": "Binary Trees",
    "avl": "AVL Trees",
    "graph": "Graph Theory",
    "spanning tree": "Spanning Trees",
    "complexity": "Time & Space Complexity",
    "big o": "Asymptotic Notation",
    "asymptotic": "Asymptotic Notation",
    "recursion": "Recursion",
    "linked list": "Linked Lists",
    "stack": "Stack",
    "queue": "Queue",
    "array": "Arrays",
    "matrix": "Matrix Operations",
    "knapsack": "Knapsack Problem",
    "travelling salesman": "Travelling Salesman Problem",
    "tsp": "Travelling Salesman Problem",
    "minimum spanning": "Minimum Spanning Tree",
    "shortest path": "Shortest Path Algorithms",
    "topological": "Topological Sorting",
    "np": "NP-Completeness",
    "p and np": "P vs NP",
    "approximation": "Approximation Algorithms",
    "branch and bound": "Branch and Bound",
    "mst": "Minimum Spanning Tree",
    "lcs": "Longest Common Subsequence",
    "optimal": "Optimization Problems",
    # OS
    "scheduling": "CPU Scheduling",
    "deadlock": "Deadlock",
    "semaphore": "Semaphores & Synchronization",
    "paging": "Memory Paging",
    "segmentation": "Memory Segmentation",
    "virtual memory": "Virtual Memory",
    "process": "Process Management",
    "thread": "Threads & Multithreading",
    "memory management": "Memory Management",
    "file system": "File Systems",
    "interrupt": "Interrupts",
    # Networks
    "tcp": "TCP/IP",
    "ip": "IP Addressing",
    "routing": "Routing Algorithms",
    "osi": "OSI Model",
    "dns": "DNS",
    "http": "HTTP Protocol",
    "socket": "Socket Programming",
    # DBMS
    "normalization": "Normalization",
    "sql": "SQL",
    "transaction": "Transactions & ACID",
    "acid": "Transactions & ACID",
    "indexing": "Indexing",
    "join": "Joins in SQL",
    "er diagram": "ER Diagrams",
    "relational": "Relational Model",
    # General CS
    "compiler": "Compiler Design",
    "automata": "Theory of Automata",
    "turing": "Turing Machine",
    "finite automata": "Finite Automata",
    "context free": "Context Free Grammar",
    "grammar": "Formal Grammar",
    "encryption": "Cryptography",
    "cryptography": "Cryptography",
    "machine learning": "Machine Learning",
    "neural network": "Neural Networks",
    "artificial intelligence": "Artificial Intelligence",
    "operating system": "Operating Systems",
    "database": "Database Management",
    "network": "Computer Networks",
    "software engineering": "Software Engineering",
    "uml": "UML Diagrams",
    "object oriented": "Object Oriented Programming",
    "inheritance": "Object Oriented Programming",
    "polymorphism": "Object Oriented Programming",
    "design pattern": "Design Patterns",
}

def clean_question_text(text):
    """Aggressively clean a question string."""
    # Remove newlines, tabs
    text = re.sub(r'[\n\r\t]+', ' ', text)
    # Remove question numbering: Q1, 1., (a), iv., Part A etc
    text = re.sub(r'(?i)^(?:Q(?:uestion)?\s*\d+[a-z]?[\s\.)\-:]*|\d+[a-z]?[\s\.)\-:]+|[ivxlIVXL]+[\s\.)\-:]+|\([a-z]\)\s*)', '', text)
    # Remove Bloom's taxonomy / CO markers
    text = re.sub(r'(?i)\s*[\(\[]?\b(?:BTL?|CO|L|PO|PSO|RBT)[1-9]\b[\)\]]?\s*', ' ', text)
    # Remove marks indicators: (10 marks), [5M], (5+5), (2×3)
    text = re.sub(r'(?i)\s*[\(\[]\s*\d+\s*(?:[×xX\+]\s*\d+\s*)*\s*(?:marks?|m|M)?\s*[\)\]]\s*$', '', text)
    text = re.sub(r'\s*[\(\[]\s*\d+\s*[\)\]]\s*$', '', text)
    # Remove PART headers
    text = re.sub(r'(?i)\bPART\s*[A-E]\b.*', '', text)
    # Remove module/unit prefixes
    text = re.sub(r'(?i)^(?:module|unit|chapter|section)\s*\d+[:\-\s]*', '', text)
    # Remove OCR noise: isolated digits, lone symbols
    text = re.sub(r'(?<!\w)\d{1,2}(?!\w)', '', text)
    text = re.sub(r'[^\w\s.,?()\-:;\'\"]+', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove leading/trailing punctuation
    text = text.strip('.,:-')
    return text.strip()

def assign_topic(question_text, fallback_counter=None):
    """
    Match question to a technical topic using the taxonomy map.
    Returns a clean topic name.
    """
    q_lower = question_text.lower()
    
    # 1. Direct match against taxonomy (longest match wins)
    best_match = None
    best_len = 0
    for keyword, topic in TECH_TOPIC_MAP.items():
        if keyword in q_lower and len(keyword) > best_len:
            best_match = topic
            best_len = len(keyword)
    
    if best_match:
        return best_match
    
    # 2. spaCy noun chunk fallback - find most technical term
    doc = nlp(question_text.lower())
    candidates = []
    stopwords_extra = {'question', 'answer', 'explain', 'describe', 'define',
                       'list', 'discuss', 'write', 'note', 'example', 'following',
                       'diagram', 'paper', 'marks', 'section', 'part', 'unit',
                       'module', 'give', 'state', 'show', 'prove', 'find',
                       'calculate', 'derive', 'obtain', 'brief', 'short', 'long'}
    
    for chunk in doc.noun_chunks:
        text_chunk = re.sub(r'^(the|a|an|this|that|these|those)\s+', '', chunk.text.strip())
        if (chunk.root.is_alpha and
            len(text_chunk) > 3 and
            len(text_chunk.split()) <= 4 and
            not chunk.root.is_stop and
            chunk.root.pos_ in ['NOUN', 'PROPN'] and
            text_chunk.lower() not in stopwords_extra):
            
            score = fallback_counter.get(text_chunk, 0) if fallback_counter else 0
            candidates.append((score, len(text_chunk), text_chunk))
    
    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return candidates[0][2].title()
    
    return "General"

def analyze_question_papers(file_paths, output_report_path=None):
    """
    Analyzes question papers to group questions by precise technical topics.
    """
    all_raw_questions = []
    overall_topic_counter = Counter()

    # Step 1: Extract all questions from all files
    for path in file_paths:
        if path.endswith('.txt'):
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            text = extract_text_from_pdf(path)

        questions = identify_questions(text)
        all_raw_questions.extend(questions)

        # Build a global noun frequency counter for fallback topic assignment
        doc = nlp(text.lower())
        for chunk in doc.noun_chunks:
            text_chunk = re.sub(r'^(the|a|an)\s+', '', chunk.text.strip())
            if (chunk.root.is_alpha and len(text_chunk) > 2 and
                len(text_chunk.split()) <= 3 and not chunk.root.is_stop and
                chunk.root.pos_ in ['NOUN', 'PROPN']):
                overall_topic_counter[text_chunk] += 1

    # Step 2: Clean questions and assign topics
    topic_question_map = defaultdict(list)  # topic -> list of cleaned question texts
    question_occurrence = Counter()

    for q_text in all_raw_questions:
        cleaned = clean_question_text(q_text)

        # Filter junk
        if len(cleaned) < 12:
            continue
        alnum = sum(1 for c in cleaned if c.isalnum())
        if alnum / len(cleaned) < 0.55:
            continue

        topic = assign_topic(cleaned, overall_topic_counter)
        key = cleaned.lower()
        question_occurrence[key] += 1
        topic_question_map[topic].append(cleaned)

    # Step 3: Deduplicate within each topic using Jaccard similarity
    result = []
    for topic, questions in topic_question_map.items():
        unique = {}  # normalized_key -> {text, occurrence_count}
        for q in questions:
            key = q.lower()
            words = set(key.split())
            
            matched = False
            for existing_key in list(unique.keys()):
                existing_words = set(existing_key.split())
                union = existing_words | words
                inter = existing_words & words
                similarity = len(inter) / len(union) if union else 0
                if similarity > 0.65:
                    unique[existing_key]['occurrence_count'] += 1
                    matched = True
                    break
            
            if not matched:
                unique[key] = {
                    "text": q,
                    "occurrence_count": question_occurrence[key]
                }

        if unique:
            result.append({
                "topic": topic,
                "questions": list(unique.values()),
                "frequency": len(unique),
                "importance_score": sum(q['occurrence_count'] for q in unique.values())
            })

    # Sort topics by importance
    result.sort(key=lambda x: x['importance_score'], reverse=True)
    return result