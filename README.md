# 🧠 AI Resume Analyzer — Professional Dashboard

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

An intelligent, AI-powered **Resume Analyzer** built with **Streamlit**, **NLP**, and **Machine Learning**.  
It evaluates resumes against job descriptions, extracts skills, generates feedback, and provides a complete professional dashboard for analysis and insights.

---

## 🚀 Features

✅ **Smart Resume Parsing**  
- Extracts text from `.pdf`, `.docx`, and image files (OCR support using Tesseract).  
- Automatically detects education, experience, and skills.

✅ **Skill Extraction & Categorization**  
- Identifies both technical and soft skills using **spaCy NLP** and a curated keyword database.  
- Groups skills into Programming, Frameworks, Databases, Tools, Visualization, and Concepts.

✅ **AI-Powered Scoring System**  
- Compares resumes with one or multiple **Job Descriptions (JDs)**.  
- Calculates **TF-IDF** and **Semantic (SBERT)** similarity.  
- Combines with weighted skill scoring to give realistic job-fit scores.

✅ **AI Feedback (OpenAI)**  
- Generates concise feedback and actionable improvement suggestions.  
- Creates improved resume summaries and project highlights using GPT models (optional).

✅ **Interactive Dashboard**  
- Built with **Streamlit** for an elegant, professional UI.  
- Sidebar navigation: *Home / Analyze / History / About*.  
- Summary metrics, charts, and expandable tabs for detailed insights.

✅ **Reports & Exports**  
- Download detailed reports as **JSON** or **PDF**.  
- Visualize skill match and category coverage with Matplotlib charts.  
- Stores history locally in **SQLite**.

---

## 🧰 Tech Stack

| Component | Technology |
|------------|-------------|
| **Frontend** | Streamlit |
| **NLP** | spaCy, Sentence Transformers |
| **Machine Learning** | scikit-learn (TF-IDF) |
| **Database** | SQLite |
| **Visualization** | Matplotlib |
| **AI Integration** | OpenAI API (optional) |
| **File Handling** | PyMuPDF, python-docx, pytesseract, Pillow |

---

## ⚙️ Installation

### 1️⃣ Clone this repository
```bash
git clone https://github.com/<your-username>/AI-Resume-Analyzer.git
cd AI-Resume-Analyzer
