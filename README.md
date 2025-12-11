# 📘 Syllabus Mapping Agent

**AI-Powered Curriculum Comparison System**

**A full-stack application that compares two curriculum files (CSV/XLSX), aligns topics across educational boards, and provides an interactive preview + downloadable results.**

### This project consists of:

- A custom-built Web UI (HTML + CSS + JavaScript) with a glassmorphism theme

- A FastAPI backend that processes file uploads

- A CurriculumComparator Python engine that performs text processing, embeddings, similarity scoring, clustering, and topic alignment

- A CSV export module for downstream analysis

**This README will help new contributors understand the architecture, setup steps, and code flow.**

## 🌐 1. Project Architecture
frontend/

 ├── index.html        # Core interface 
 
 ├── style.css         # Complete UI theme (glassmorphism, neon highlights, layout)
 
 ├── script.js         # Form logic, FastAPI calls, dynamic UI updates
 
backend/

 ├── main.py           # FastAPI server
 
 ├── curriculum_comparator.py   # Core AI matching pipeline
 
 ├── requirements.txt  # Dependencies
 
output/

 └── comparison_output.csv      # Final exported match results

## 💡 2. How the System Works (High-Level)

#### Step 1: User Provides Inputs

User enters:

- Board A Name

- Board B Name

- Uploads CSV/XLSX files for both boards

#### Step 2: Frontend Sends a Request

- script.js creates a FormData() object and POSTs it to:

- POST http://localhost:5000/process

#### Step 3: FastAPI Receives Files

- FastAPI:

- Saves uploaded files temporarily

- Instantiates the CurriculumComparator

- Executes the full comparison pipeline

- Produces:

- A preview (first 10 rows)

- A Base64-encoded CSV of full results

#### Step 4: Results Sent Back to Frontend

- Frontend displays:

- Progress bar animation

- Preview table with horizontal + vertical scroll

- Full CSV download button

#### Step 5: User Downloads Results

- The final output is:

  - Curriculum_Comparison_Results.csv

## 🧠 3. The AI Matching Pipeline

The CurriculumComparator internally performs:

#### 1. Data Loading

- Reads CSV/XLSX for both boards.

#### 2. Preprocessing

- Cleans text (lowercasing, removing special characters, normalizing).

#### 3. Combined Text Construction

- Merges related columns into a single textual representation per topic.

#### 4. Embedding Generation

- Uses a sentence embedding model (such as MiniLM, BERT, etc.)

#### 5. Similarity Matrix Computation

- Cosine similarity between embeddings of Board A and Board B topics.

#### 6. Topic-Level Matching

- Selects the top matches based on thresholds + ranking logic.

#### 7. Concept Extraction (Optional)

- Extracts keywords or concepts using NLP techniques.

#### 8. Clustering

- Helps group similar topics for better cross-board alignment.

#### 9. Export

- Writes the output match matrix and scores to a CSV file.

## ⚙️ 4. Setting Up the Project
**Backend Setup**

#### 1. Create a virtual environment
- python -m venv venv

#### 2. Activate it

- Windows

  - venv\Scripts\activate


- Mac/Linux

  - source venv/bin/activate

#### 3. Install backend dependencies
- pip install -r backend/requirements.txt

#### 4. Start the FastAPI server
- cd backend
- uvicorn main:app --reload --port 5000


**Your backend is now running at:**

- http://localhost:5000

## 🎨 5. Running the Frontend

Simply open:

frontend/index.html

in your web browser.

No framework required.

#### The frontend uses:

**Pure HTML**

**CSS Glassmorphism UI**

**JavaScript for UX + API calls**


## 📤 6. File Format Requirements

#### Your uploaded CSV/XLSX must contain:

- Course Name

- Chapter/Topic Name

- Grade/Class

- Description (optional)

**Must remove:**

- Serial Number

- Topic ID

- Subject ID

- Chapter ID

**This ensures the AI model receives clean, meaningful text inputs.**

## 📄 7. API Contract
POST /process
Request (multipart/form-data)
boardA: string
boardB: string
fileA: file (CSV/XLSX)
fileB: file (CSV/XLSX)



## 🚀 8. How to Extend the Project

- Friendly to new contributors.

- Add new similarity metrics

**Update:**

- curriculum_comparator.py → compute_topic_similarity()

- Add new models

- Replace embedding model in:

  - generate_embeddings()

 -Add authentication

- Wrap FastAPI endpoints with JWT.

- Add support for PDFs

## 🙌 9. Techstack

**Developed with:**

- FastAPI

- JavaScript Fetch API

- Custom-designed UI (Glassmorphism + Neon)

- NLP + Embedding Models
