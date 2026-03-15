# Financial Report Analyst — Phase 1 (Qdrant version)

A complete RAG pipeline that reads financial PDFs and answers natural language questions.
Uses **Qdrant** (via Docker) as the vector database — no C++ compiler needed.

---

## What you need on your PC

| Tool | Purpose | Status |
|---|---|---|
| Python 3.11.5 | Run the code | Already installed |
| Docker Desktop | Run Qdrant vector DB | Already installed |
| OpenAI API key | Embeddings + GPT-4o | Need to create |

---

## Step 1 — Start Qdrant in Docker

Open CMD and run:
```bash
docker run -d --name qdrant -p 6333:6333 -v C:\Users\LENOVO\qdrant_data:/qdrant/storage qdrant/qdrant
```

Verify it's running:
```bash
docker ps
```

Open dashboard in browser:
```
http://localhost:6333/dashboard
```

> Every time you restart your PC — open Docker Desktop first, then run:
> `docker start qdrant`

---

## Step 2 — Project setup

```bash
# Create virtual environment with Python 3.11
py -3.11 -m venv venv

# Activate it
venv\Scripts\activate

# Install all packages (no C++ needed)
pip install -r requirements.txt
```

---

## Step 3 — Add your OpenAI API key

```bash
copy .env.example .env
```

Open `.env` in Notepad and replace the placeholder:
```
OPENAI_API_KEY=sk-your-actual-key-here
```

Get your key from: https://platform.openai.com/api-keys

---

## How to run

### Ingest a financial PDF
```bash
python financial_analyst.py --ingest hdfc_q3_2024.pdf --company HDFC --quarter Q3 --year 2024
```

### Ask a question
```bash
python financial_analyst.py --ask "What is the NPA ratio?"
```

### Ask with filter (narrow search to specific company or quarter)
```bash
python financial_analyst.py --ask "What is the gross NPA?" --company HDFC --quarter Q3
python financial_analyst.py --ask "What are the risk factors?" --company ICICI --year 2024
```

### See all ingested documents
```bash
python financial_analyst.py --list
```

---

## Where to get financial PDFs to test with

All free and publicly available:

- **BSE India** — bseindia.com → search company → Financials → Annual Reports
- **NSE India** — nseindia.com → same path
- **HDFC Bank** — search "HDFC Bank investor relations quarterly results"
- **ICICI Bank** — icicibank.com → Investor Relations → Annual Reports
- **SBI** — sbi.co.in → Investor Relations

Good first test: download any bank's Q3 2024 quarterly earnings PDF (usually 20-40 pages).

---

## Example questions to try

```bash
python financial_analyst.py --ask "What is the gross NPA ratio?"
python financial_analyst.py --ask "What was the net interest income?"
python financial_analyst.py --ask "What are the key risk factors mentioned?"
python financial_analyst.py --ask "What is the capital adequacy ratio?"
python financial_analyst.py --ask "How did the loan book grow compared to last quarter?"
python financial_analyst.py --ask "What is the return on equity?"
python financial_analyst.py --ask "Summarize the management commentary"
```

---

## Project structure

```
financial-analyst/
├── financial_analyst.py   ← complete pipeline (Qdrant version)
├── requirements.txt       ← all dependencies
├── .env.example           ← API key template
└── .env                   ← your actual API key (never commit this to git)
```

Note: No local DB folder needed — Qdrant stores all vectors inside Docker,
persisted to C:\Users\LENOVO\qdrant_data on your PC.

---

## How the pipeline works

```
PDF file
   │
   ▼
pdfplumber          Extract text page by page + convert tables to text
   │
   ▼
Chunker             Split into 500-word overlapping chunks
                    Each chunk tagged: company · quarter · year · page
   │
   ▼
OpenAI Embeddings   Convert each chunk to a 1536-number vector
(text-embedding-3-small)
   │
   ▼
Qdrant (Docker)     Store vectors + text + metadata
   │
   ▼  (on query)
User Question       Also converted to a vector
   │
   ▼
Qdrant Search       Find 5 most semantically similar chunks
   │
   ▼
GPT-4o              Answer using only those 5 chunks + cite sources
   │
   ▼
Answer + Citations
```

---

## Common errors and fixes

**ModuleNotFoundError: No module named 'openai'**
→ Virtual environment not activated. Run `venv\Scripts\activate`

**openai.AuthenticationError**
→ API key is wrong or missing. Check your .env file.

**Cannot connect to Qdrant**
→ Qdrant container is not running. Run `docker start qdrant`
→ If first time: `docker run -d --name qdrant -p 6333:6333 qdrant/qdrant`

**No relevant chunks found**
→ You haven't ingested any documents yet. Run `--ingest` first.

**PDF extraction returns empty text**
→ PDF is scanned (image-based). pdfplumber can't read images.
→ Use a PDF with selectable text — most modern bank reports work fine.
