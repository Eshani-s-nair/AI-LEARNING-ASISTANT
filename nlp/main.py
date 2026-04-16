from nlp_pipeline import get_summary_and_keywords, improve_summary_with_ollama, show_summary_in_browser
from db import DatabaseManager
from flashcard import generate_flashcards, run_flashcards
from quiz_mod import generate_fib, generate_tf, generate_mcq, generate_msq
from run_quiz import run_quiz

if __name__ == "__main__":
    print("\n🚀 PrepMate is starting...")
    file_path = input("Enter file path (PDF or DOCX): ")

    # 1. NLP PROCESSING
    print("\n--- Analyzing Document ---")
    try:
        summary_sentences, keywords = get_summary_and_keywords(file_path)
        print("✅ Analysis complete.")
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        input("\nPress Enter to exit...")
        exit()

    # 1.5. AI REFINEMENT
    print("\n--- Refining with AI ---")
    refined_summary = ""
    try:
        refined_summary = improve_summary_with_ollama(summary_sentences)
        print("✅ AI refinement complete.")
    except Exception:
        print("⚠️ Ollama not running or error. Skipping refinement.")

    # 2. SAVE TO DATABASE
    print("\n--- Saving to Database ---")
    db = DatabaseManager()
    # NOTE: user_id is hardcoded as this script doesn't handle user sessions.
    # In a real web application, this would be the ID of the logged-in user.
    user_id = 1
    db.save_data(file_path, summary_sentences, keywords, user_id=user_id)
    
    # 2.5 GENERATE FRONTEND DATA (Creates summary_data.js)
    # This function generates 'summary_data.js' in the front folder
    show_summary_in_browser(summary_sentences, keywords, refined_summary)
    
    # 3. CHOOSE STUDY MODE
    print("\n" + "="*20 + " STUDY OPTIONS " + "="*20)
    print("[1] Flashcards")
    print("[2] Quiz")
    try:
        choice = input("Select an option (1 or 2): ")
        if choice == "1":
            print("\n--- Generating Flashcards ---")
            cards = generate_flashcards(summary_sentences)
            run_flashcards(cards)
        elif choice == "2":
            print("\n--- Generating Quiz ---")
            fib = generate_fib(keywords, summary_sentences)
            tf = generate_tf(summary_sentences, keywords)
            mcq = generate_mcq(summary_sentences, keywords)
            msq = generate_msq(summary_sentences, keywords)
            run_quiz(fib, tf, mcq, msq)
        else:
            print("❌ Invalid choice. Returning to main menu.")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        
    input("\nFinished. Press Enter to exit...")