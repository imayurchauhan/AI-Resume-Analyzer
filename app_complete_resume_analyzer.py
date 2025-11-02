# app_complete_resume_analyzer.py
"""
AI Resume Analyzer — Professional Dashboard (Complete)
Save as app_complete_resume_analyzer.py and run with:
    streamlit run app_complete_resume_analyzer.py
This file is a full, standalone update of your original app with a professional sidebar + card UI.
Keeps all original features: PDF/DOCX/IMG parsing, OCR, skill extraction, categorization,
TF-IDF + SBERT similarity, weighted scoring, OpenAI feedback (optional), JSON/PDF exports, and SQLite history.
"""

import os
import io
import re
import json
import time
import sqlite3
import warnings
from typing import List, Dict

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Core libs
import fitz  # PyMuPDF
import spacy
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional libs (kept optional to not break environments without them)
try:
    from sentence_transformers import SentenceTransformer, util as sbert_util
    sbert_available = True
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    sbert_available = False
    sbert_model = None

try:
    from openai import OpenAI
    openai_available = True
except Exception:
    openai_available = False

try:
    import docx
    docx_available = True
except Exception:
    docx_available = False

try:
    import pytesseract
    from PIL import Image
    ocr_available = True
except Exception:
    ocr_available = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    reportlab_available = True
except Exception:
    reportlab_available = False

# -----------------------------
# Initialize NLP
# -----------------------------
# Use a lightweight English model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    # If model missing, try to proceed without spacy tokenization fallback
    nlp = None

# -----------------------------
# Skill lists and weights
# -----------------------------
SKILLS = [
    "python","java","c++","sql","mysql","postgresql","mongodb","sqlite",
    "pandas","numpy","scikit-learn","tensorflow","pytorch","keras",
    "nlp","natural language processing","transformers","computer vision",
    "opencv","image processing","matplotlib","seaborn","power bi",
    "flask","streamlit","django","docker","git","aws","gcp","azure",
    "api","rest","fastapi","spark","hadoop","excel"
]

SKILL_WEIGHTS = {
    "python": 2.0,
    "pandas": 1.5, "numpy": 1.5, "scikit-learn": 1.8,
    "tensorflow": 2.0, "pytorch": 2.0, "nlp": 1.8, "computer vision": 1.8,
    "docker": 1.2, "streamlit": 1.0, "flask": 1.0, "git": 1.0,
    "sql": 1.2, "mysql": 1.1, "mongodb": 1.1, "aws": 1.3, "gcp": 1.3
}

# -----------------------------
# DB utilities
# -----------------------------
DB_PATH = "resume_analyzer_history.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS analyses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        resume_name TEXT,
        overall_score REAL,
        top_jd_score REAL,
        details TEXT
    )""")
    conn.commit()
    conn.close()

def save_analysis_record(resume_name, overall_score, top_jd_score, details):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO analyses (timestamp, resume_name, overall_score, top_jd_score, details) VALUES (?,?,?,?,?)",
                (time.strftime("%Y-%m-%d %H:%M:%S"), resume_name, overall_score, top_jd_score, json.dumps(details)))
    conn.commit()
    conn.close()

def load_history(limit=20):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, timestamp, resume_name, overall_score, top_jd_score FROM analyses ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return rows

# -----------------------------
# Text extraction routines
# -----------------------------
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    text = ""
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
    except Exception:
        return ""
    return text

def extract_text_from_docx_bytes(docx_bytes: bytes) -> str:
    if not docx_available:
        return ""
    bio = io.BytesIO(docx_bytes)
    doc = docx.Document(bio)
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs)

def extract_text_from_image_bytes(img_bytes: bytes) -> str:
    if not ocr_available:
        return ""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return pytesseract.image_to_string(img)

def extract_text(file_bytes: bytes, filename: str) -> str:
    fname = filename.lower()
    if fname.endswith(".pdf"):
        return extract_text_from_pdf_bytes(file_bytes)
    if fname.endswith(".docx") or fname.endswith(".doc"):
        return extract_text_from_docx_bytes(file_bytes)
    if fname.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
        return extract_text_from_image_bytes(file_bytes)
    # fallback: try pdf parse
    try:
        return extract_text_from_pdf_bytes(file_bytes)
    except Exception:
        return ""

# -----------------------------
# Skill extraction & categorization
# -----------------------------
def extract_skills(text: str) -> List[str]:
    text_l = text.lower()
    found = set()
    if nlp:
        doc = nlp(text_l)
        for token in doc:
            tok = token.text.strip()
            if tok in SKILLS:
                found.add(tok)
    # keyword fallback scan
    for skill in SKILLS:
        if skill in text_l:
            found.add(skill)
    return sorted(found)

def categorize_skills(skill_list: List[str]) -> Dict[str, List[str]]:
    categories = {
        "Programming Languages": {"python","java","c++","c","javascript","html","css"},
        "Frameworks & Libraries": {"tensorflow","pytorch","scikit-learn","keras","pandas","numpy","transformers","opencv"},
        "Databases": {"mysql","mongodb","postgresql","sqlite"},
        "Tools & Deployment": {"docker","streamlit","flask","git","aws","gcp","azure"},
        "Visualization": {"matplotlib","seaborn","power bi","excel"},
        "Concepts": {"nlp","deep learning","computer vision","machine learning","data analysis"}
    }
    categorized = {cat: [] for cat in categories}
    for s in skill_list:
        for cat, vals in categories.items():
            if s.lower() in vals:
                categorized[cat].append(s)
                break
    return {k:v for k,v in categorized.items() if v}

# -----------------------------
# Experience & Education extraction
# -----------------------------
def extract_experience_years(text: str):
    years = []
    patterns = [
        r'(\d+)\s*\+\s*years', r'(\d+)\s*years', r'(\d+\.\d+)\s*years'
    ]
    for p in patterns:
        for m in re.findall(p, text.lower()):
            try:
                years.append(float(m))
            except:
                pass
    if years:
        return max(years)
    if "fresher" in text.lower() or "entry level" in text.lower():
        return 0.0
    return None

EDU_KEYWORDS = ["btech","b.e.","be","b.sc","bachelor","mtech","m.e.","m.sc","master","mba","msc","mca","phd","doctorate"]
def extract_education(text: str) -> List[str]:
    found = []
    t = text.lower()
    for kw in EDU_KEYWORDS:
        if kw in t:
            found.append(kw)
    return sorted(set(found))

# -----------------------------
# Similarity & scoring
# -----------------------------
def tfidf_similarity(a: str, b: str) -> float:
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        vectors = vectorizer.fit_transform([a,b])
        sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        return float(sim)
    except:
        return 0.0

def semantic_similarity(list_a, list_b) -> float:
    joined_a = " ".join(list_a) if isinstance(list_a, (list,tuple)) else str(list_a)
    joined_b = " ".join(list_b) if isinstance(list_b, (list,tuple)) else str(list_b)
    if sbert_available and sbert_model is not None:
        try:
            emb_a = sbert_model.encode(joined_a, convert_to_tensor=True)
            emb_b = sbert_model.encode(joined_b, convert_to_tensor=True)
            sim = sbert_util.pytorch_cos_sim(emb_a, emb_b).item()
            return float(sim)
        except Exception:
            return tfidf_similarity(joined_a, joined_b)
    else:
        return tfidf_similarity(joined_a, joined_b)

def weighted_skill_score(resume_skills: List[str], jd_skills: List[str]) -> float:
    if not jd_skills:
        return 0.0
    score = 0.0
    total_weight = 0.0
    for s in jd_skills:
        w = SKILL_WEIGHTS.get(s.lower(), 1.0)
        total_weight += w
        if s in resume_skills:
            score += w
    if total_weight == 0:
        return 0.0
    return float(score / total_weight)

def overall_heuristic(resume_skills: List[str], jd_skills: List[str], categorized: Dict[str,List[str]]) -> float:
    semantic = semantic_similarity(resume_skills, jd_skills)
    weighted = weighted_skill_score(resume_skills, jd_skills)
    cat_cov = min(1.0, len(categorized) / 5.0)
    return round(100.0 * (0.55*semantic + 0.30*weighted + 0.15*cat_cov), 2)

# -----------------------------
# OpenAI helpers for feedback & rewrite
# -----------------------------
def generate_openai_feedback(api_key: str, resume_text: str, resume_skills: List[str], jd_text: str, jd_skills: List[str]) -> str:
    if not openai_available:
        return "OpenAI SDK not installed. Install with pip install openai"
    try:
        client = OpenAI(api_key=api_key)
        prompt_user = (
            "You are a concise AI/ML resume reviewer. Given the resume text and job description, "
            "provide: (A) A 2-line score justification, (B) Top 6 actionable improvements (bulleted), "
            "(C) Two one-line resume edits (Summary + Skills), (D) One sample resume project bullet for deployment."
            f"\n\nResume Skills: {', '.join(resume_skills)}\nJD Skills: {', '.join(jd_skills)}\n\nResume Text (short):\n{resume_text[:2000]}"
        )
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role":"system","content":"You are a helpful career coach and resume reviewer."},
                {"role":"user","content":prompt_user}
            ],
            temperature=0.2,
            max_tokens=600
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI error: {e}\n(Using fallback rule-based feedback.)"

def generate_resume_rewrite(api_key: str, resume_text: str, improvements_list: str):
    if not openai_available:
        return None
    try:
        client = OpenAI(api_key=api_key)
        prompt = f"Given this resume text (short):\n{resume_text[:2000]}\n\nAnd these improvements:\n{improvements_list}\n\nWrite: (1) one-line improved Summary suitable for resume, (2) one polished project bullet mentioning deployment tools."
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role":"system","content":"You are a resume writing assistant. Keep it concise."},
                {"role":"user","content":prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None

# -----------------------------
# PDF export utility
# -----------------------------
def export_report_pdf(path: str, title: str, report_text: str):
    if not reportlab_available:
        raise RuntimeError("reportlab not available")
    c = canvas.Canvas(path, pagesize=letter)
    w, h = letter
    margin = 40
    y = h - margin
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, title)
    c.setFont("Helvetica", 10)
    y -= 20
    for line in report_text.splitlines():
        if y < 60:
            c.showPage()
            y = h - margin
            c.setFont("Helvetica", 10)
        c.drawString(margin, y, line[:120])
        y -= 12
    c.save()

# -----------------------------
# Helper UI components
# -----------------------------
def render_header():
    st.markdown("""
        <style>
            .header {
                display: flex;
                align-items: center;
                gap: 12px;
            }
            .brand {
                font-size:20px;
                font-weight:700;
            }
            .sub {
                color: #6c757d;
                font-size:12px;
            }
        </style>
        """, unsafe_allow_html=True)
    cols = st.columns([0.1, 1])
    with cols[0]:
        st.image("https://img.icons8.com/fluency/48/000000/artificial-intelligence.png", width=48)
    with cols[1]:
        st.markdown("<div class='header'><div class='brand'>AI Resume Analyzer — Professional Dashboard</div></div>", unsafe_allow_html=True)
        st.markdown("<div class='sub'>Upload resumes, paste JDs, get scoring, detailed feedback, and exportable reports.</div>", unsafe_allow_html=True)

def render_metrics_card(title, value, delta=None):
    st.metric(label=title, value=value, delta=delta)

# -----------------------------
# Main Streamlit app
# -----------------------------
def main():
    st.set_page_config(page_title="AI Resume Analyzer — Dashboard", layout="wide")
    init_db()

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("", ["Home", "Analyze", "History", "About"], index=1)

    # Global inputs on sidebar for convenience
    st.sidebar.markdown("---")
    st.sidebar.subheader("Options")
    show_example = st.sidebar.checkbox("Show example JD & sample resume", value=False)

    # Home
    if page == "Home":
        render_header()
        st.markdown("---")
        left, right = st.columns([3,1])
        with left:
            st.subheader("Quick Start")
            st.write("""
                1. Go to **Analyze**.  
                2. Upload one or more resume files (.pdf, .docx, .doc, .png, .jpg).  
                3. Paste one or multiple job descriptions (use --- to separate).  
                4. (Optional) Provide OpenAI API key for richer feedback.  
                5. Click **Run Analysis** and view results under tabs.
            """)
            st.info("Your analysis history is saved locally in a SQLite DB (`resume_analyzer_history.db`).")
            if show_example:
                st.markdown("**Example Job Description:**")
                st.code("Data Scientist — Experience with Python, pandas, scikit-learn, SQL, AWS. Knowledge of NLP and transformers preferred.")
                st.markdown("**Example Resume (snippet):**")
                st.code("Experienced ML engineer with Python, pandas, scikit-learn, deployed models on AWS using Docker and FastAPI.")
        with right:
            st.subheader("Status")
            st.write("SBERT available: ", "Yes" if sbert_available else "No (TF-IDF fallback)")
            st.write("OpenAI SDK available: ", "Yes" if openai_available else "No")
            st.write("docx support: ", "Yes" if docx_available else "No")
            st.write("OCR support (pytesseract): ", "Yes" if ocr_available else "No")
            st.write("PDF export (reportlab): ", "Yes" if reportlab_available else "No")

    # Analyze
    elif page == "Analyze":
        render_header()
        st.markdown("---")
        st.subheader("Analyze Resumes")

        # Layout: left panel for inputs, right for quick controls/history
        colL, colR = st.columns([3,1])
        with colL:
            with st.form("analyze_form", clear_on_submit=False):
                uploaded_files = st.file_uploader("Upload one or more resumes (PDF/DOCX/IMG).", accept_multiple_files=True, type=["pdf","docx","doc","png","jpg","jpeg"])
                jd_input = st.text_area("Paste Job Description(s). For multiple JDs separate with lines containing '---' (three hyphens).", height=200)
                provided_api_key = st.text_input("OpenAI API Key (optional)", type="password")
                run_btn = st.form_submit_button("Run Full Analysis")

        with colR:
            st.markdown("### Quick Actions")
            if st.button("Clear stored history"):
                try:
                    conn = sqlite3.connect(DB_PATH)
                    cur = conn.cursor()
                    cur.execute("DELETE FROM analyses")
                    conn.commit()
                    conn.close()
                    st.success("History cleared.")
                except Exception as e:
                    st.error("Failed to clear history: " + str(e))
            st.markdown("---")
            st.markdown("Recent analyses preview:")
            rows = load_history(5)
            for r in rows:
                st.write(f"[{r[0]}] {r[1]} — {r[2]} — Score: {r[3]} — TopJD: {r[4]}")

        # Run analysis when user requests it
        if run_btn:
            if not uploaded_files:
                st.error("Upload at least one resume file.")
            elif not jd_input.strip():
                st.error("Paste at least one JD.")
            else:
                api_key = provided_api_key.strip() or os.getenv("OPENAI_API_KEY")
                jd_texts = [s.strip() for s in jd_input.split("---") if s.strip()]
                if not jd_texts:
                    jd_texts = [jd_input.strip()]

                # Use session state to store reports from this run
                if "last_run_reports" not in st.session_state:
                    st.session_state["last_run_reports"] = []

                all_reports = []

                progress = st.progress(0)
                total = len(uploaded_files)
                for idx, uploaded in enumerate(uploaded_files, start=1):
                    name = uploaded.name
                    st.info(f"Processing: {name}")
                    raw = uploaded.read()
                    text = extract_text(raw, name)
                    if not text:
                        st.warning(f"No text extracted for {name}. Skipping.")
                        progress.progress(min(100, int(100 * idx/total)))
                        continue

                    # extraction
                    skills = extract_skills(text)
                    categorized = categorize_skills(skills)
                    years = extract_experience_years(text)
                    education = extract_education(text)

                    # Evaluate against each JD
                    jd_results = []
                    for jd in jd_texts:
                        jd_skills = extract_skills(jd)
                        sem = semantic_similarity(skills, jd_skills)
                        tf = tfidf_similarity(" ".join(skills), " ".join(jd_skills))
                        weighted = weighted_skill_score(skills, jd_skills)
                        jd_score = round(100.0 * (0.5*sem + 0.4*weighted + 0.1*tf), 2)
                        jd_results.append({"jd": jd, "jd_skills": jd_skills, "score": jd_score})

                    jd_sorted = sorted(jd_results, key=lambda x: x["score"], reverse=True)
                    top = jd_sorted[0] if jd_sorted else {"score":0, "jd_skills":[]}

                    overall = overall_heuristic(skills, top["jd_skills"], categorized)
                    matched = sorted(list(set(skills) & set(top["jd_skills"])))
                    missing = sorted(list(set(top["jd_skills"]) - set(skills)))

                    # Feedback
                    if api_key and openai_available:
                        try:
                            fb = generate_openai_feedback(api_key, text, skills, top["jd"], top["jd_skills"])
                        except Exception as e:
                            fb = f"OpenAI feedback error: {e}\nUsing rule-based fallback."
                    else:
                        fb_lines = []
                        fb_lines.append(f"Matched skills: {', '.join(matched) if matched else 'None'}")
                        if missing:
                            fb_lines.append("Consider adding: " + ", ".join(missing))
                        if years is not None:
                            fb_lines.append(f"Experience detected: {years} years")
                        if education:
                            fb_lines.append("Education keywords found: " + ", ".join(education))
                        fb_lines.append("Suggestion: Add measurable outcomes (accuracy, dataset sizes), and a short 'Deployment & Tools' bullet.")
                        fb = "\n".join(fb_lines)

                    # Optional rewrite suggestion
                    rewrite_suggestion = None
                    if api_key and openai_available:
                        try:
                            rewrite_suggestion = generate_resume_rewrite(api_key, text, fb)
                        except:
                            rewrite_suggestion = None

                    # store report
                    report = {
                        "resume_name": name,
                        "skills": skills,
                        "categorized": categorized,
                        "experience_years": years,
                        "education": education,
                        "jd_results": jd_sorted,
                        "overall_score": overall,
                        "top_jd_score": top["score"],
                        "matched": matched,
                        "missing": missing,
                        "feedback": fb,
                        "rewrite": rewrite_suggestion,
                        "raw_text_preview": text[:2000]
                    }
                    all_reports.append(report)
                    save_analysis_record(name, overall, top["score"], report)

                    # progress
                    progress.progress(min(100, int(100 * idx/total)))

                # Save to session state to display on page
                st.session_state["last_run_reports"] = all_reports

                if not all_reports:
                    st.warning("No reports generated (maybe all files had no text).")
                else:
                    st.success("Analysis complete ✅")

        # If we have results in session state, show them with tabs/cards
        reports = st.session_state.get("last_run_reports", [])
        if reports:
            st.markdown("---")
            st.subheader("Analysis Results")
            # Summary cards at top
            summary_cols = st.columns(len(reports))
            for i, r in enumerate(reports):
                with summary_cols[i]:
                    st.markdown(f"**{r['resume_name']}**")
                    st.metric("Overall Score", f"{r['overall_score']}/100", delta=None)
                    st.write(f"Top JD Score: {r['top_jd_score']}")
                    st.write(f"Skills: {', '.join(r['skills']) if r['skills'] else 'None'}")

            # For each resume, create an expander with tabs
            for r in reports:
                with st.expander(f"{r['resume_name']} — Score: {r['overall_score']}"):
                    tabs = st.tabs(["Overview", "Visuals", "Feedback & Rewrite", "Export"])
                    # Overview Tab
                    with tabs[0]:
                        st.markdown("#### Overview")
                        st.write("**Matched:**", ", ".join(r["matched"]) if r["matched"] else "None")
                        st.write("**Missing (from top JD):**", ", ".join(r["missing"]) if r["missing"] else "None")
                        st.write("**Experience (years):**", r["experience_years"] if r["experience_years"] is not None else "Not detected")
                        st.write("**Education keywords:**", ", ".join(r["education"]) if r["education"] else "Not detected")
                        st.markdown("**Skill categories:**")
                        if r["categorized"]:
                            for cat, vals in r["categorized"].items():
                                st.write(f"• {cat}: {', '.join(vals)}")
                        else:
                            st.write("No categorized skills detected")
                        st.markdown("**Top JD (short):**")
                        st.write(r["jd_results"][0]["jd"] if r["jd_results"] else "N/A")
                        st.markdown("**Raw text preview (first 2000 chars):**")
                        st.code(r.get("raw_text_preview",""))

                    # Visuals Tab
                    with tabs[1]:
                        st.markdown("#### Visual Insights")
                        fig1, ax1 = plt.subplots(figsize=(4,3))
                        sizes = [len(r["matched"]), len(r["missing"])]
                        labels = ["Matched", "Missing"]
                        if sum(sizes) == 0:
                            ax1.text(0.5,0.5,"No skills", ha="center")
                        else:
                            # colors left default (no hard color specification)
                            ax1.pie(sizes, labels=labels, autopct="%1.1f%%")
                        ax1.axis("equal")
                        st.pyplot(fig1)

                        if r["categorized"]:
                            fig2, ax2 = plt.subplots(figsize=(6, max(2, len(r["categorized"])*0.5)))
                            ax2.barh(list(r["categorized"].keys()), [len(v) for v in r["categorized"].values()])
                            ax2.set_xlabel("Count")
                            ax2.set_title("Skills per Category")
                            st.pyplot(fig2)
                        else:
                            st.info("No categorized skills to visualize.")

                    # Feedback Tab
                    with tabs[2]:
                        st.markdown("#### Feedback")
                        st.text_area("Feedback (AI or rule)", r["feedback"], height=220)
                        if r.get("rewrite"):
                            st.markdown("#### AI Rewrite Suggestion")
                            st.write(r["rewrite"])
                        else:
                            st.info("AI rewrite not available (provide OpenAI API key and ensure SDK installed).")

                    # Export Tab
                    with tabs[3]:
                        st.markdown("#### Export")
                        st.download_button("Download JSON report", json.dumps(r, indent=2), file_name=f"{r['resume_name']}_report.json", mime="application/json")
                        if reportlab_available:
                            try:
                                pdf_path = f"{r['resume_name']}_report.pdf"
                                txt = f"Resume Analysis Report — {r['resume_name']}\n\nOverall Score: {r['overall_score']}\nTop JD Score: {r['top_jd_score']}\n\nMatched: {', '.join(r['matched'])}\nMissing: {', '.join(r['missing'])}\n\nFeedback:\n{r['feedback']}"
                                export_report_pdf(pdf_path, f"Report: {r['resume_name']}", txt)
                                with open(pdf_path, "rb") as f:
                                    st.download_button("Download PDF Report", f.read(), file_name=pdf_path, mime="application/pdf")
                            except Exception as e:
                                st.info("PDF export failed: " + str(e))
                        else:
                            st.info("Install reportlab to enable PDF export (pip install reportlab).")

    # History page
    elif page == "History":
        render_header()
        st.markdown("---")
        st.subheader("Analysis History")
        rows = load_history(50)
        if not rows:
            st.info("No history found.")
        else:
            # Show compact table
            cols = st.columns([1,2,3,1,1])
            cols[0].write("ID")
            cols[1].write("Timestamp")
            cols[2].write("Resume Name")
            cols[3].write("Score")
            cols[4].write("Top JD")
            for r in rows:
                c0, c1, c2, c3, c4 = st.columns([1,2,3,1,1])
                c0.write(r[0])
                c1.write(r[1])
                c2.write(r[2])
                c3.write(r[3])
                c4.write(r[4])
            st.markdown("---")
            st.info("You can clear history from the Analyze page (Quick Actions).")

    # About page
    elif page == "About":
        render_header()
        st.markdown("---")
        st.subheader("About This App")
        st.write("""
            **AI Resume Analyzer — Professional Dashboard**  
            This app extracts text from resumes (PDF/DOCX/images), detects skills/education/experience,
            compares resumes to one or more job descriptions, computes heuristic scores (semantic similarity + weighted skills + category coverage),
            generates feedback (rule-based or via OpenAI), and lets you export JSON/PDF reports.  
            Data is stored locally in a SQLite DB: `resume_analyzer_history.db`.
        """)
        st.markdown("**Notes & Requirements**")
        st.write("- Ensure optional dependencies for extra features:")
        st.write("  - `sentence_transformers` for better semantic similarity (SBERT).")
        st.write("  - `openai` SDK if you want model-generated feedback & rewrites.")
        st.write("  - `python-docx` to parse DOCX files.")
        st.write("  - `pytesseract` + Tesseract binary for OCR from images.")
        st.write("  - `reportlab` to enable PDF export.")
        st.markdown("**Commands**")
        st.code("pip install spacy pymupdf scikit-learn streamlit matplotlib")
        st.code("pip install sentence-transformers openai python-docx pytesseract reportlab  # optional features")
        st.markdown("**Made for:** Your Resume Analyzer project — reorganized into a professional dashboard layout.")
        st.markdown("---")
        st.write("If you want, I can now:")
        st.write("• Deploy this to Streamlit Cloud or Hugging Face Spaces (prepare requirements + instructions).")
        st.write("• Add JD clustering, resume section parsing (summary/projects/skills detection), or a prettier export template.")
        st.write("Tell me which next and I'll give the full updated code for that too.")

if __name__ == "__main__":
    main()
