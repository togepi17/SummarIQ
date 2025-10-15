import os
import re
import json
import PyPDF2
import markdown
from flask import Flask, render_template, request, redirect, url_for, flash, session
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

app = Flask(__name__)
app.secret_key = "supersecretkey"  # For sessions and flash messages

# -----------------------------
# Configuration and Initialization
# -----------------------------
os.environ["GOOGLE_API_KEY"] = ""
llm = GoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

# -----------------------------
# File Extraction Functions
# -----------------------------
def extract_text_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def process_file(file_path):
    if not os.path.exists(file_path):
        return None, "File does not exist."
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        return extract_text_from_txt(file_path), None
    elif ext == ".pdf":
        return extract_text_from_pdf(file_path), None
    else:
        return None, "Unsupported file format. Please use TXT or PDF."

# -----------------------------
# LangChain Utility: Text Splitting
# -----------------------------
def split_text(text, chunk_size=1000, overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)

# -----------------------------
# Summarization Function (Markdown → HTML)
# -----------------------------
def summarize_text(text, method="easy"):
    """
    Generate a detailed summary in Markdown format.
    The Markdown will include:
      - Headings (using '#' for titles)
      - Bold text (using '**')
      - Bullet lists (using '-' for lists)
      - A section called "Points to Remember"
    For the 80/20 method, the summary includes separate sections for "Key 20%" and "Supporting 80%".
    """
    prompts = {
        "easy": (
            "Generate a detailed note summary of the following text in Markdown format. "
            "Include a title, bullet points for key points, and a section titled 'Points to Remember'.\n\n{text}"
        ),
        "80/20": (
            "Analyze the following text and extract the most critical 20% of the content (Key 20%) that represents 80% of the ideas (Supporting 80%). "
            "Present your answer in Markdown with two sections: one titled 'Key 20%' and one titled 'Supporting 80%'. "
            "Also include bullet points and a section 'Points to Remember'.\n\n{text}"
        ),
        "understanding": (
            "Rewrite the following text into an easily understandable set of notes in Markdown format. "
            "Use headings, bullet points for key takeaways, and include a section 'Points to Remember'.\n\n{text}"
        )
    }
    if method not in prompts:
        return "Invalid summarization method."
    
    prompt_template = PromptTemplate(template=prompts[method], input_variables=["text"])
    chunks = split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]
    
    chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=prompt_template, combine_prompt=prompt_template)
    markdown_summary = chain.run(docs)
    
    # Convert Markdown to HTML for display
    html_summary = markdown.markdown(markdown_summary)
    return html_summary

# -----------------------------
# Quiz Generation Function (Plain Text with Explanation)
# -----------------------------
def generate_quiz(summary):
    """
    Generate a multiple-choice quiz with 5 questions (2 easy, 2 medium, 1 hard)
    based on the provided summary.
    
    Instruct the LLM to output the quiz in plain text with the following exact format for each question:
    
    Question <number> [<difficulty>]:
    Question: <question text>
    Option A: <option text>
    Option B: <option text>
    Option C: <option text>
    Option D: <option text>
    Answer: <correct letter>
    Explanation: <explanation text>
    
    Each question is separated by a blank line.
    """
    quiz_prompt = (
        "Based on the following summary, generate a multiple-choice quiz with 5 questions: 2 easy, 2 medium, and 1 hard. "
        "For each question, follow this exact format:\n\n"
        "Question <number> [<difficulty>]:\n"
        "Question: <question text>\n"
        "Option A: <option text>\n"
        "Option B: <option text>\n"
        "Option C: <option text>\n"
        "Option D: <option text>\n"
        "Answer: <correct letter>\n"
        "Explanation: <explanation text>\n\n"
        "Separate each question by a blank line.\n\n"
        "Summary:\n{text}"
    )
    prompt_template = PromptTemplate(template=quiz_prompt, input_variables=["text"])
    quiz_chain = LLMChain(llm=llm, prompt=prompt_template)
    raw_quiz = quiz_chain.run({"text": summary})
    return parse_quiz_text(raw_quiz)

def parse_quiz_text(quiz_text):
    """
    Parse the quiz text (with explanation) into a list of question dictionaries.
    Expected format for each question (8 lines):
    
    Line 1: Question <number> [<difficulty>]:
    Line 2: Question: <question text>
    Line 3: Option A: <option text>
    Line 4: Option B: <option text>
    Line 5: Option C: <option text>
    Line 6: Option D: <option text>
    Line 7: Answer: <correct letter>
    Line 8: Explanation: <explanation text>
    """
    questions = []
    # Split by two or more newlines
    parts = re.split(r"\n\s*\n", quiz_text.strip())
    for part in parts:
        lines = part.strip().splitlines()
        if len(lines) < 8:
            continue  # Skip incomplete questions
        header_match = re.match(r"Question\s+\d+\s+\[(\w+)\]:", lines[0].strip())
        if not header_match:
            continue
        difficulty = header_match.group(1).lower()
        q_match = re.match(r"Question:\s*(.+)", lines[1].strip())
        if not q_match:
            continue
        question_text = q_match.group(1)
        options = {}
        for line in lines[2:6]:
            opt_match = re.match(r"Option\s+([A-D]):\s*(.+)", line.strip())
            if opt_match:
                options[opt_match.group(1)] = opt_match.group(2)
        ans_match = re.match(r"Answer:\s*([A-D])", lines[6].strip())
        if not ans_match:
            continue
        correct_answer = ans_match.group(1)
        expl_match = re.match(r"Explanation:\s*(.+)", lines[7].strip())
        explanation = expl_match.group(1) if expl_match else ""
        questions.append({
            "difficulty": difficulty,
            "question": question_text,
            "options": options,
            "correct_answer": correct_answer,
            "explanation": explanation
        })
    return questions

# -----------------------------
# Flask Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    method_descriptions = {
        "easy": "A simple, concise summary in note form.",
        "80/20": "Extracts the critical 20% (Key 20%) and supporting 80% as notes.",
        "understanding": "Rewrites content into clear, understandable notes."
    }
    
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part in the request.")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("No file selected.")
            return redirect(request.url)
        filename = file.filename
        os.makedirs("uploads", exist_ok=True)
        file_path = os.path.join("uploads", filename)
        file.save(file_path)
        
        method = request.form.get("method")
        text, error = process_file(file_path)
        if error:
            flash(error)
            return redirect(request.url)
        
        summary = summarize_text(text, method)
        session["summary"] = summary  # Store for quiz generation
        return render_template("result.html", summary=summary)
    
    return render_template("index.html", method_descriptions=method_descriptions)

@app.route("/quiz", methods=["GET", "POST"])
def quiz():
    summary = session.get("summary", None)
    if not summary:
        flash("No summary available. Please generate a summary first.")
        return redirect(url_for("index"))
    
    quiz_data = generate_quiz(summary)
    if not quiz_data:
        flash("Quiz generation failed. Please try again.")
        return redirect(url_for("index"))
    
    session["quiz"] = quiz_data  # Store quiz for answer checking
    return render_template("quiz.html", quiz=quiz_data)

@app.route("/submit_quiz", methods=["POST"])
def submit_quiz():
    quiz_data = session.get("quiz", [])
    score = 0
    results = []
    for i, q in enumerate(quiz_data):
        user_ans = request.form.get(f"q{i}")
        correct = q.get("correct_answer")
        is_correct = (user_ans == correct)
        if is_correct:
            score += 1
        # Include correct option text and explanation in the result
        correct_text = q.get("options", {}).get(correct, "")
        results.append({
            "question": q.get("question"),
            "selected": user_ans,
            "correct": correct,
            "correct_text": correct_text,
            "explanation": q.get("explanation", ""),
            "is_correct": is_correct
        })
    return render_template("quiz_result.html", score=score, total=len(quiz_data), results=results)

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)
