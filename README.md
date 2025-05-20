# **Unlocking the Potential of Arabic NLP: High-Quality Dataset and Preprocessing Framework for Arabic Large Language Models**

This project aims to build a **high-quality Arabic dataset for low-resource LLMs** focused on **academic content** while providing a **web-based preprocessing framework** called **Nukhba Labs**. The system **collects, cleans, annotates, fine-tunes, and evaluates** Arabic text for training **Arabic Large Language Models (LLMs)** with minimal manual effort.

---

### 📌 **Project Status**
- ✅ Data collection, cleaning, and annotation completed  
- ✅ Initial LLM fine-tuning using AraBERT  
- ✅ Web-based framework deployed (Firebase + Flask)  
- ✅ Evaluation using grammar, lexical, readability, and topic metrics  
- ⏳ Final LLM integration in progress  

---

## 🔁 **End-to-End Pipeline**

1. **Data Collection**
   - Source: Common Crawl, Arabic Datasets  
   - 🔧 Script: `Download_WARC/download_warc.py`

2. **Data Cleaning**
   - Includes: Tokenization, HTML Cleaning, Deduplication, Blocklist Filtering  
   - 🔧 Script: `CommonCrawl_Pipeline/pipeline.py`

3. **Annotation & Scoring**
   - Scored from 0–5 for academic relevance using **ALLaM-7B-Instruct**  
   - 🔧 Script: `Testing_LLM/test.py`

4. **Model Fine-Tuning**
   - Annotated subset used to fine-tune **AraBERT**, which then auto-scores the full dataset  
   - 🔧 Script: `Academic_Specific/Academic.py`

5. **Evaluation**
   - Evaluates processed datasets using:
     - ✅ Grammar scoring (LLM-based)
     - ✅ Lexical diversity metrics
     - ✅ Readability analysis
     - ✅ Topic distribution via BERTopic  
   - 🔧 Script: `Evaluation/Evaluate.py`  
   - 🧠 SLURM Jobs: `run_evaluate.sh`

6. **LLM Integration**
   - Final step: Fine-tune or integrate academic dataset into a large Arabic LLM  
   - 🚧 In progress

---

## 🌐 **Nukhba Labs Web Framework**

A full-stack Arabic dataset preprocessing framework.

### 🔑 **Key Features**
- Firebase Auth (email, Google, GitHub)
- Drag & drop dataset upload
- Selectable cleaning steps (tokenize, normalize, remove noise, etc.)
- Output format selector (CSV, JSON, XLSX)
- Dataset history and download
- Backend powered by Flask (Render)

### 📁 Web Files
| Page | Function |
|------|----------|
| `index.html` | Landing page with login + explainer video |
| `login.html` | Secure login (email + social auth) |
| `signup.html` | Signup linked to Firebase |
| `dashboard.html` | View history of cleaned datasets |
| `upload.html` | Upload interface with cleaning options |
| `preview.html` | Download confirmation screen |
| `firebase.js` | Firebase Auth + Firestore config |
| `app.py` | Flask backend for cleaning |
| `render.yaml` | Render deployment config |
| `requirements.txt` | Python backend dependencies |

---

## 🛠️ **Technologies Used**

- **Transformers** – ALLaM-7B-Instruct, AraBERT, LLaMA-3 via Hugging Face  
- **DeepSpeed + SLURM** – HPC model training  
- **Firebase Auth** – Google, GitHub, Email login  
- **Flask** – Python backend for file processing  
- **BERTopic** – Topic modeling on academic data  
- **TextStat, NLTK** – Readability & lexical metrics  
- **HTML/CSS/JS** – Full frontend UI  
- **Render** – Deploy backend API  
- **GitHub Pages** – Deploy frontend  


---

## 🔗 Live Demo

- **Frontend:** [https://ameeraattiah.github.io/nukhba-labs](https://ameeraattiah.github.io/nukhba-labs)  
- **Backend:** Deployed via Render 


