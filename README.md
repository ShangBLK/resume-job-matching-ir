# Resume-Job Matching IR System

This project explores how well traditional and modern information retrieval techniques can match resumes to relevant job descriptions. Built for CSC 575: Intelligent Information Retrieval at DePaul University, it compares a TF-IDF-based retrieval model against a semantic retrieval model using pre-trained language embeddings.

## Objective

- Match resumes with appropriate job postings
- Compare two retrieval methods:
  - **TF-IDF** (term frequency–inverse document frequency)
  - **LLM embeddings** via Sentence Transformers
- Analyze performance through:
  - Top-k job retrievals
  - Score distributions
  - Overlap metrics

## Project Structure


```text
resume-job-matching-ir/
├── data/
│   ├── Resume.csv                 # Resume dataset
│   ├── job_title_des.csv          # Job descriptions
│
├── notebooks/
│   ├── shared_setup.ipynb         # Preprocessing + cleaning
│   ├── tfidf_retrieval.ipynb      # TF-IDF model and results
│   ├── semantic_llm_retrieval.ipynb # LLM model and results
│   ├── compare_models.ipynb       # Overlap & scoring comparison
│   ├── tfidf_resume_to_jobs.csv   # TF-IDF output file
│   └── llm_resume_to_jobs.csv     # LLM output file
│
├── pdfs/
│   └── project_paper.pdf          # (Optional) project report

├── .gitignore
└── README.md

## Setup Instructions

### Requirements

- Python 3.8+
- Jupyter Notebook
- Required packages:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `sentence-transformers`
  - `pymupdf`

### Installation

If using pip:

pip install -U sentence-transformers pymupdf pandas matplotlib

cpp
Copy
Edit

If using conda:

conda install pip -y
pip install -U sentence-transformers pymupdf pandas matplotlib

markdown
Copy
Edit

## How to Run

Run the following notebooks in this order:

1. `shared_setup.ipynb`  
   Loads and preprocesses both resume and job datasets.

2. `tfidf_retrieval.ipynb`  
   Uses TF-IDF vectorization and cosine similarity to match resumes to job descriptions.

3. `semantic_llm_retrieval.ipynb`  
   Applies SentenceTransformer (MiniLM) to compute semantic similarity between resumes and job descriptions.

4. `compare_models.ipynb`  
   Loads both output `.csv` files, computes overlap in top-k results, and visualizes comparison metrics.

## Outputs

Each retrieval notebook produces a CSV file:

- `tfidf_resume_to_jobs.csv`  
- `llm_resume_to_jobs.csv`

The final comparison notebook (`compare_models.ipynb`) performs:

- Resume-by-resume overlap comparison between both models
- Average similarity score analysis
- Bar graph of job match agreement

## Key Techniques

- **TF-IDF + Cosine Similarity**: baseline keyword matching
- **SentenceTransformer Embeddings**: context-aware retrieval using `all-MiniLM-L6-v2`
- **PyMuPDF**: used for extracting raw text from PDF resumes
- **Matplotlib**: used for visualizing retrieval overlap

## License

This repository was developed for educational use only as part of a graduate AI course at DePaul University. Reuse is permitted for academic, instructional, or non-commercial purposes with credit.

## Author

**Shang Andrews**  
Graduate Student, Artificial Intelligence  
DePaul University  
GitHub: [ShangBLK](https://github.com/ShangBLK)
