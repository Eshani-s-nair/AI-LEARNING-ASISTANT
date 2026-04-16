import json
import os
import webbrowser

def run_quiz(fib, tf, mcq, msq):
    """
    Combines quiz data, saves it to a JS file, and opens the quiz HTML page.
    """
    # 1. Combine all questions into a single dictionary
    quiz_data = {
        "fill_in_the_blank": fib,
        "true_false": tf,
        "mcq": mcq,
        "msq": msq
    }

    # 2. Determine paths relative to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    front_dir = os.path.join(base_dir, "..", "front")
    data_file = os.path.join(front_dir, "quiz_data.js")
    html_file = os.path.join(front_dir, "quiz.html")

    # 3. Save quiz data as a JS variable
    # The variable name 'generatedQuizData' must match what quiz.html expects
    js_content = f"const generatedQuizData = {json.dumps(quiz_data, indent=4)};"

    # 4. Write to file and open in browser
    try:
        with open(data_file, "w", encoding="utf-8") as f:
            f.write(js_content)
        print(f"\n✅ Quiz data generated and saved to {data_file}")
        print("🚀 Opening Quiz in browser...")
        webbrowser.open('file://' + os.path.realpath(html_file))
    except Exception as e:
        print(f"❌ Error saving quiz data: {e}")