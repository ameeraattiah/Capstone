# **Arabic NLP Dataset Processing Pipeline**  

This project focuses on **processing large-scale Arabic text data** from **Common Crawl** and other sources to create **high-quality datasets** for training Arabic **Large Language Models (LLMs)**. The pipeline **cleans, filters, annotates, fine-tunes, and evaluates** the dataset before integration with existing LLMs.  

### 📌 **Project Status:**  
   - ✅ Data collection, cleaning, and annotation are complete
   - ✅ Fine-tuning AraBERT is in progress
   - ⏳ *Upcoming: AI-based evaluation & LLM integration*
---

## **Pipeline Overview**  
Our data processing pipeline follows these main steps:

1. **Data Collection**
   - Sources: **Common Crawl, Web Scraping, Arabic Datasets**  
   - **Relevant Scripts:** `Download_WARC/download_warc.py`  

2. **Data Cleaning** 
   - Tasks: **Text Cleaning, Blocklist Filtering, Deduplication, ...**  
   - **Relevant Scripts:** `CommonCrawl_Pipeline/pipeline.py`  

3. **Annotation Scoring** 
   - Classifies text as **Academic or Not Academic** from 0 to 5.
   - Uses **LLaMA-3.1-8B** for scoring  
   - **Relevant Scripts:** `Testing_LLM/test.py`  

4. **Fine-Tuning**  
   - Trains **AraBERT** on the **annotated subset**  
   - **Relevant Scripts:** `Academic_Specific/Academic.py`  

5. **AI-Based Evaluation**  
   - Benchmarks: **COPA-ar, HellaSwag-ar, PIQA-ar**  
   - **Relevant Scripts:** Upcoming...

6. **Integration with LLMs**  
   - Final step: Use the processed data to **fine-tune an LLM or integrate with an existing model**  
   - **Relevant Scripts:** Upcoming...
---

## **Repository Structure & Script Descriptions**  

### **1. `Download_WARC/download_warc.py`** *(Step 1: Data Collection)*
**Function:** Downloads **WARC files** from **Common Crawl** that contain **Arabic content** before storing them.

**📌 How it Works:**
- Reads a list of **Common Crawl WARC file paths**.
- **Checks if the file contains Arabic content** before downloading.
- Skips already processed files to prevent duplicates.
---

### **2. `CommonCrawl_Pipeline/pipeline.py`** *(Step 2: Data Cleaning)*
**Function:** Processes raw **WARC files** by extracting, cleaning, filtering, and deduplicating Arabic text.  

**📌 How it Works:**
- **Extract & Clean:** Extracts text from **WARC files**, removes HTML tags, special characters, and normalizes Arabic.  
- **Filter & Refine:** Removes **non-Arabic, spam, and low-quality text** using blocklists and quality checks.  
- **Deduplicate & Enrich:** Eliminates **duplicates with MinHash** and adds metadata like language score and token count.  
- **Store & Prepare:** Saves the **final high-quality dataset** in JSON format for Arabic NLP model training.  
---

### **3. `Testing_LLM/test.py`**  *(Step 3: Testing Annotation & Scoring)*
**Function:**  This codes main purpose was to test **different LLM models** to determine which one works best with our **annotation instructions** and to **validate prompt effectiveness** before full integration with the main code `Academic_Specific/Academic.py`. 

**📌 How it Works:**
- Loads **LLaMA-3.1-8B** to process text.
- **Generates a score (1-5) based on academic quality**.
- Uses **structured prompt** to ensure accurate evaluation.
- Stores **annotated text** for later fine-tuning.
---

### **4. `Academic_Specific/Academic.py`** *(Step 4: Annotation & Fine-Tuning)*  

**Function:** This script first **annotates 5,000 samples from Common Crawl using LLaMA-3** and then **fine-tunes AraBERT** to automatically filter and score the remaining **995,000 samples**. The final output is a **JSON file containing only high-quality academic content**, scored **3 and above**.  

**📌 How it Works:**  
- Uses **LLaMA-3** to annotate an initial **5,000 samples** for academic relevance.  
- Fine-tunes **AraBERT** based on these annotations for **automated filtering**.  
- **AraBERT then classifies and filters** the remaining **995,000 samples**.  
- Saves the final dataset as a **JSON file containing only academic content (score ≥ 3)**.  
- Runs on **Ibex HPC** with **DeepSpeed & SLURM** for large-scale processing.  

---

#### **📌 Key Technologies Used**
- **Transformers (Hugging Face)** – LLaMA & AraBERT  
- **DeepSpeed** – Efficient model training  
- **SLURM** – Parallel job execution on KAUST'S Ibex HPC  
- **FastText & NLTK** – Language detection & tokenization  
- **WandB** – Experiment tracking  

---