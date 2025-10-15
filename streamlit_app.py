import os
import streamlit as st
from app import process_file, summarize_text, generate_quiz

# --------------------
# 🔐 Set your Gemini API Key (Secure method recommended)
# --------------------

# ✅ Preferred: via environment variable
# os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# 🚨 Quick test only (local only): Hardcoded API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyAMFqmjvKtM6s3B9fbdy2dMziV8jplXhDQ"

# --------------------
# UI Settings
# --------------------
st.set_page_config(page_title="SummarIQ", layout="centered")
st.title("📄 SummarIQ: Smart Summarizer + Quiz Generator")

# --------------------
# File Upload + Summarization
# --------------------
uploaded_file = st.file_uploader("📂 Upload a TXT or PDF file", type=["txt", "pdf"])
method = st.selectbox("🧠 Select summarization method", ["easy", "80/20", "understanding"])

if uploaded_file:
    with open("tempfile", "wb") as f:
        f.write(uploaded_file.read())
    
    text, error = process_file("tempfile")
    if error:
        st.error(error)
    else:
        if st.button("✨ Generate Summary"):
            summary_html = summarize_text(text, method)
            st.markdown("### 📝 Generated Summary", unsafe_allow_html=True)
            st.markdown(summary_html, unsafe_allow_html=True)
            st.session_state["summary"] = summary_html
            st.session_state["raw_summary"] = text
            st.session_state["show_quiz_button"] = True

# --------------------
# Quiz Generation
# --------------------
if st.session_state.get("show_quiz_button"):
    if st.button("🧠 Quiz Me!"):
        quiz_data = generate_quiz(st.session_state["raw_summary"])
        st.session_state["quiz_data"] = quiz_data
        st.session_state["show_quiz"] = True

# --------------------
# Quiz Display + Submission
# --------------------
if st.session_state.get("show_quiz"):
    st.markdown("## 📋 Quiz Section")
    quiz_data = st.session_state["quiz_data"]
    user_answers = {}

    for i, q in enumerate(quiz_data):
        st.markdown(f"**Q{i+1} ({q['difficulty'].capitalize()})**: {q['question']}")
        options = q["options"]
        user_answers[i] = st.radio(
            f"Your answer for Q{i+1}:", 
            list(options.keys()), 
            format_func=lambda x: f"{x}. {options[x]}",
            key=f"quiz_q{i}"
        )
        st.markdown("---")

    if st.button("✅ Submit Quiz"):
        score = 0
        for i, q in enumerate(quiz_data):
            selected = user_answers[i]
            correct = q["correct_answer"]
            if selected == correct:
                score += 1
                st.success(f"✅ Q{i+1}: Correct! ({correct} - {q['options'][correct]})")
            else:
                st.error(f"❌ Q{i+1}: Incorrect. Correct is {correct} - {q['options'][correct]})")
                st.info(f"**Explanation:** {q['explanation']}")
        
        st.markdown(f"## 🏁 Final Score: **{score} / {len(quiz_data)}**")


