"""
Financial Report Analyst — Phase 1
=====================================
RAG pipeline using:
  - NVIDIA NIM  → LLM (answering questions)   — free API credits
  - Ollama      → Embeddings (nomic-embed-text) — free, runs locally
  - Qdrant      → Vector database               — free, runs in Docker

Prerequisites:
  1. Qdrant running in Docker:
       docker run -d --name qdrant -p 6333:6333 qdrant/qdrant

  2. Ollama installed with embedding model:
       ollama pull nomic-embed-text

  3. NVIDIA API key in .env file:
       NVIDIA_API_KEY=nvapi-...

Run:
  python financial_analyst.py --ingest report.pdf --company HDFC --quarter Q3 --year 2024
  python financial_analyst.py --ask "What is the NPA ratio?"
  python financial_analyst.py --list
"""

import os
import sys
import argparse
import ollama
import pdfplumber

from openai import OpenAI
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

# ─── Setup ────────────────────────────────────────────────────────────────────

load_dotenv()  # reads all values from your .env file

console = Console()

# ── NVIDIA NIM client ──
# NVIDIA uses the same API format as OpenAI — we just point
# the OpenAI client to NVIDIA's server URL instead.
# This means all openai_client.chat.completions.create() calls
# go to NVIDIA's Llama model, not GPT-4o.
NVIDIA_API_KEY  = os.getenv("NVIDIA_API_KEY")
NVIDIA_LLM_MODEL = os.getenv("NVIDIA_LLM_MODEL", "meta/llama-3.1-70b-instruct")

nvidia_client = OpenAI(
    api_key=NVIDIA_API_KEY,
    base_url="https://integrate.api.nvidia.com/v1"  # NVIDIA NIM endpoint
)

# ── Ollama embedding model ──
# nomic-embed-text produces 768-dimensional vectors
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# ── Qdrant client ──
qdrant = QdrantClient(host="localhost", port=6333)

COLLECTION_NAME = "financial_reports"
VECTOR_SIZE     = 768   # nomic-embed-text output size (Ollama)


def ensure_collection_exists():
    """
    Creates the Qdrant collection if it doesn't already exist.
    COSINE distance = find vectors with similar meaning/direction.
    Vector size must match the embedding model output (768 for nomic-embed-text).
    """
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        console.print(f"[green]✓[/green] Created Qdrant collection: '{COLLECTION_NAME}'")
    else:
        console.print(f"[cyan]Using existing Qdrant collection: '{COLLECTION_NAME}'[/cyan]")


# ─── STEP 1: PDF Extraction ───────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Opens a PDF and extracts text + tables page by page.

    Why page by page?
      We tag each chunk with its page number for citations.
      When the LLM answers "Gross NPA was 1.26%" it can also say
      "Source: HDFC Q3 2024 Page 14" — that's what makes it trustworthy.

    Returns: [{"page": 1, "content": "..."}, {"page": 2, ...}, ...]
    """
    pages = []

    console.print(f"\n[cyan]Opening PDF:[/cyan] {pdf_path}")

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        console.print(f"[cyan]Total pages:[/cyan] {total_pages}")

        for i, page in enumerate(pdf.pages):
            page_content = ""

            # Extract regular paragraph text
            text = page.extract_text()
            if text:
                page_content += text

            # Extract tables (balance sheets, NPA tables, ratio tables etc.)
            # Converts table cells into pipe-separated readable text
            tables = page.extract_tables()
            for table in tables:
                if table:
                    page_content += "\n" + table_to_text(table)

            if page_content.strip():
                pages.append({
                    "page":    i + 1,
                    "content": page_content.strip()
                })

            if (i + 1) % 10 == 0:
                console.print(f"  Processed {i + 1}/{total_pages} pages...")

    console.print(f"[green]✓[/green] Extracted text from {len(pages)} pages")
    return pages


def table_to_text(table: list) -> str:
    """
    Converts a 2D list (table) into a pipe-separated readable string.

    Input:  [["Metric", "Q3", "Q2"], ["Gross NPA", "1.26%", "1.34%"]]
    Output: "Metric | Q3 | Q2\nGross NPA | 1.26% | 1.34%"

    This lets the LLM read table data as plain text.
    """
    rows = []
    for row in table:
        clean_row = [str(cell).strip() if cell else "" for cell in row]
        rows.append(" | ".join(clean_row))
    return "\n".join(rows)


# ─── STEP 2: Chunking ─────────────────────────────────────────────────────────

def split_into_chunks(pages: list[dict], chunk_size: int = 500, overlap: int = 100) -> list[dict]:
    """
    Splits page text into smaller overlapping chunks.

    Why chunk at all?
      LLMs have a context limit — we can't send a 100-page PDF.
      Instead we find the 5 most relevant 500-word chunks and send only those.

    Why overlap?
      If an important sentence sits at the boundary of two chunks,
      the overlap (100 words) ensures it appears fully in at least one chunk.

    chunk_size = 500 words → good for financial text
    overlap    = 100 words → 20% overlap is standard practice
    """
    chunks   = []
    chunk_id = 0

    for page_data in pages:
        words    = page_data["content"].split()
        page_num = page_data["page"]

        start = 0
        while start < len(words):
            end        = start + chunk_size
            chunk_text = " ".join(words[start:end])

            if len(chunk_text.strip()) > 50:   # skip meaningless tiny chunks
                chunks.append({
                    "id":   f"chunk_{chunk_id}",
                    "text": chunk_text,
                    "page": page_num
                })
                chunk_id += 1

            start += chunk_size - overlap  # slide forward keeping the overlap

    console.print(
        f"[green]✓[/green] Created {len(chunks)} chunks "
        f"(chunk_size={chunk_size} words, overlap={overlap} words)"
    )
    return chunks


# ─── STEP 3: Embedding (Ollama) ───────────────────────────────────────────────

def get_embedding(text: str) -> list[float]:
    """
    Converts text into a 768-dimensional vector using Ollama's nomic-embed-text.

    This vector captures the *meaning* of the text numerically.
    Similar meaning → vectors point in a similar direction.

    Example:
      "What is NPA ratio?"            → [0.12, -0.45, 0.89, ...]
      "Gross non-performing assets %"  → [0.11, -0.43, 0.91, ...]
      → these two are very close in 768D space → semantic search finds them

    Ollama runs locally — no API call, no cost, no internet needed.
    """
    response = ollama.embeddings(
        model=OLLAMA_EMBED_MODEL,
        prompt=text
    )
    return response["embedding"]


def embed_in_batches(texts: list[str], batch_size: int = 50) -> list[list[float]]:
    """
    Embeds a large list of texts one by one (Ollama doesn't batch).

    batch_size=50 means we log progress every 50 chunks.
    Ollama is fast locally — typically 0.1-0.3s per chunk.
    """
    all_embeddings = []

    for i, text in enumerate(texts):
        embedding = get_embedding(text)
        all_embeddings.append(embedding)

        if (i + 1) % 50 == 0:
            console.print(f"  Embedded {i + 1}/{len(texts)} chunks...")

    console.print(f"  Embedded {len(texts)}/{len(texts)} chunks...")
    return all_embeddings


# ─── STEP 4: Store in Qdrant ──────────────────────────────────────────────────

def store_in_qdrant(chunks, embeddings, company, quarter, year, pdf_path):
    """
    Stores chunk text + embedding vectors in Qdrant.

    Each stored item ("Point") has 3 parts:
      id      → unique integer (hashed from chunk id + metadata)
      vector  → 768 numbers (the embedding)
      payload → the original text + metadata (company, quarter, year, page)

    Why upsert?
      Safe to re-run ingestion — updates existing chunks instead of duplicating.
    """
    points = []

    for chunk, embedding in zip(chunks, embeddings):
        point_id = abs(hash(
            f"{company}_{quarter}_{year}_{chunk['id']}"
        )) % (2 ** 63)

        points.append(PointStruct(
            id=point_id,
            vector=embedding,
            payload={
                "text":    chunk["text"],
                "company": company.upper(),
                "quarter": quarter.upper(),
                "year":    str(year),
                "page":    str(chunk["page"]),
                "source":  os.path.basename(pdf_path)
            }
        ))

    # Upload in batches of 100
    batch_size = 100
    for i in range(0, len(points), batch_size):
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=points[i: i + batch_size]
        )
        console.print(
            f"  Stored {min(i + batch_size, len(points))}/{len(points)} chunks in Qdrant..."
        )

    console.print(
        f"[green]✓[/green] Stored {len(chunks)} chunks "
        f"[{company.upper()} · {quarter.upper()} · {year}]"
    )


# ─── STEP 5: Retrieval ────────────────────────────────────────────────────────

def retrieve_relevant_chunks(query: str, n_results: int = 5, filters: dict = None) -> list[dict]:
    """
    Embeds the user's question and finds the most similar chunks in Qdrant.

    How it works:
      1. Convert question to a 768D vector (using Ollama)
      2. Qdrant compares it to every stored vector using cosine similarity
      3. Returns top n_results chunks — these are what we send to NVIDIA LLM

    filters (optional) — narrow down the search space first:
      {"company": "HDFC"}                    → only HDFC docs
      {"company": "HDFC", "quarter": "Q3"}   → only HDFC Q3 docs

    Score meaning:
      1.0 = perfect match
      0.8+ = very relevant
      0.5  = somewhat related
      < 0.3 = probably not useful
    """
    query_embedding = get_embedding(query)

    qdrant_filter = None
    if filters:
        conditions    = [
            FieldCondition(key=key, match=MatchValue(value=value))
            for key, value in filters.items()
        ]
        qdrant_filter = Filter(must=conditions)

    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=n_results,
        query_filter=qdrant_filter,
        with_payload=True
    )

    chunks = []
    for hit in results:
        chunks.append({
            "text":     hit.payload["text"],
            "metadata": {
                "company": hit.payload.get("company"),
                "quarter": hit.payload.get("quarter"),
                "year":    hit.payload.get("year"),
                "page":    hit.payload.get("page"),
                "source":  hit.payload.get("source"),
            },
            "score": round(hit.score, 3)
        })

    return chunks


# ─── STEP 6: Answer Generation (NVIDIA NIM) ───────────────────────────────────

def generate_answer(query: str, context_chunks: list[dict]) -> str:
    """
    Sends the question + retrieved chunks to NVIDIA's Llama 3.1 model.

    Why temperature=0.1?
      We want factual, consistent answers — not creative ones.
      Low temperature = LLM sticks close to the provided context.

    The system prompt is the most important part — it tells the LLM:
      - Only use what's in the context (prevents hallucination)
      - Always cite sources (company, quarter, year, page)
      - Say "I don't know" if info isn't in the chunks

    This is called "grounding" — critical for financial data accuracy.
    """
    # Build context block from retrieved chunks
    context = ""
    for i, chunk in enumerate(context_chunks):
        meta = chunk["metadata"]
        context += (
            f"\n--- Source {i + 1} "
            f"[{meta.get('company', '?')} · {meta.get('quarter', '?')} "
            f"{meta.get('year', '?')} · Page {meta.get('page', '?')}] "
            f"(relevance: {chunk['score']}) ---\n"
            f"{chunk['text']}\n"
        )

    system_prompt = """You are a precise financial analyst AI assistant.

Your job is to answer questions about financial reports accurately.

Rules you must strictly follow:
1. Answer ONLY using the provided context. Do not use any outside knowledge.
2. Always cite your sources — mention company, quarter, year and page number.
3. If the answer is NOT in the context, say exactly:
   "I couldn't find this information in the provided documents."
4. For numbers and ratios, quote them exactly — never round or approximate.
5. Keep answers professional, clear and concise.
"""

    user_message = f"""Context extracted from financial documents:
{context}

Question: {query}

Answer based only on the context above. Include citations."""

    response = nvidia_client.chat.completions.create(
        model=NVIDIA_LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message}
        ],
        temperature=0.1,
        max_tokens=1024
    )

    return response.choices[0].message.content


# ─── FULL PIPELINE ────────────────────────────────────────────────────────────

def ingest_document(pdf_path: str, company: str, quarter: str, year: str):
    """
    Full ingestion pipeline:
    PDF → extract → chunk → embed (Ollama) → store (Qdrant)
    """
    console.print(Panel(
        f"[bold]Ingesting document[/bold]\n\n"
        f"File    : {pdf_path}\n"
        f"Company : {company.upper()}\n"
        f"Quarter : {quarter.upper()}\n"
        f"Year    : {year}\n\n"
        f"Embedding model : {OLLAMA_EMBED_MODEL} (Ollama)\n"
        f"LLM model       : {NVIDIA_LLM_MODEL} (NVIDIA NIM)",
        style="cyan"
    ))

    ensure_collection_exists()

    # Step 1 — Extract text and tables from PDF
    pages = extract_text_from_pdf(pdf_path)

    # Step 2 — Split into overlapping chunks
    chunks = split_into_chunks(pages, chunk_size=500, overlap=100)

    # Step 3 — Embed all chunks using Ollama (local, free)
    console.print(f"\n[cyan]Generating embeddings with Ollama ({OLLAMA_EMBED_MODEL})...[/cyan]")
    texts      = [c["text"] for c in chunks]
    embeddings = embed_in_batches(texts)
    console.print(f"[green]✓[/green] Embeddings done")

    # Step 4 — Store vectors in Qdrant
    store_in_qdrant(chunks, embeddings, company, quarter, year, pdf_path)

    console.print(Panel(
        f"[bold green]Ingestion complete![/bold green]\n\n"
        f"Stored {len(chunks)} chunks for "
        f"{company.upper()} {quarter.upper()} {year}\n\n"
        f"Now ask a question:\n"
        f'  python financial_analyst.py --ask "What is the NPA ratio?"',
        style="green"
    ))


def ask_question(query: str, company: str = None, quarter: str = None, year: str = None):
    """
    Full query pipeline:
    Question → embed (Ollama) → retrieve (Qdrant) → answer (NVIDIA NIM)
    """
    console.print(Panel(f"[bold]Question:[/bold] {query}", style="cyan"))

    # Build optional filter
    filters = {}
    if company: filters["company"] = company.upper()
    if quarter: filters["quarter"] = quarter.upper()
    if year:    filters["year"]    = str(year)

    # Step 1 — Find relevant chunks from Qdrant
    console.print("\n[cyan]Searching Qdrant for relevant chunks...[/cyan]")
    chunks = retrieve_relevant_chunks(
        query,
        n_results=5,
        filters=filters if filters else None
    )

    if not chunks:
        console.print("[red]No relevant chunks found.[/red]")
        console.print("Make sure you have ingested documents first:")
        console.print(
            "  python financial_analyst.py "
            "--ingest report.pdf --company HDFC --quarter Q3 --year 2024"
        )
        return

    # Show retrieved chunks — helps you understand what the LLM will see
    console.print(f"\n[cyan]Top {len(chunks)} relevant chunks found:[/cyan]")
    for i, chunk in enumerate(chunks):
        meta = chunk["metadata"]
        console.print(
            f"  [{i + 1}] {meta.get('company')} · {meta.get('quarter')} "
            f"{meta.get('year')} · Page {meta.get('page')} "
            f"(score: {chunk['score']})"
        )

    # Step 2 — Generate answer using NVIDIA NIM
    console.print(f"\n[cyan]Generating answer with {NVIDIA_LLM_MODEL} (NVIDIA NIM)...[/cyan]")
    answer = generate_answer(query, chunks)

    console.print(Panel(
        Markdown(answer),
        title="[bold green]Answer[/bold green]",
        style="green"
    ))


def list_documents():
    """Shows all documents currently stored in Qdrant."""
    try:
        results, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=10000,
            with_payload=True,
            with_vectors=False
        )
    except Exception:
        console.print("[yellow]No documents found or Qdrant collection doesn't exist.[/yellow]")
        return

    if not results:
        console.print("[yellow]No documents ingested yet.[/yellow]")
        return

    # Group by company + quarter + year
    docs = {}
    for point in results:
        p   = point.payload
        key = f"{p.get('company')} · {p.get('quarter')} {p.get('year')}"
        docs[key] = docs.get(key, 0) + 1

    console.print("\n[bold]Documents stored in Qdrant:[/bold]")
    for doc, count in sorted(docs.items()):
        console.print(f"  {doc}  →  {count} chunks")

    console.print(f"\n[cyan]Total chunks: {len(results)}[/cyan]")
    console.print("[cyan]Dashboard   : http://localhost:6333/dashboard[/cyan]")


# ─── STARTUP CHECKS ───────────────────────────────────────────────────────────

def run_startup_checks():
    """
    Checks all 3 dependencies before running any command:
      1. NVIDIA API key is set in .env
      2. Qdrant Docker container is reachable
      3. Ollama is running with the embedding model available
    """
    all_ok = True

    # Check 1 — NVIDIA API key
    if not NVIDIA_API_KEY:
        console.print("[red]ERROR: NVIDIA_API_KEY not found in .env file[/red]")
        console.print("Get your free key from: https://build.nvidia.com")
        all_ok = False

    # Check 2 — Qdrant
    try:
        qdrant.get_collections()
    except Exception:
        console.print("[red]ERROR: Cannot connect to Qdrant[/red]")
        console.print("Start it with: docker start qdrant")
        console.print("Or first time: docker run -d --name qdrant -p 6333:6333 qdrant/qdrant")
        all_ok = False

    # Check 3 — Ollama
    try:
        available_models = [m["name"] for m in ollama.list()["models"]]
        embed_model_base = OLLAMA_EMBED_MODEL.split(":")[0]
        model_found = any(embed_model_base in m for m in available_models)

        if not model_found:
            console.print(f"[red]ERROR: Ollama model '{OLLAMA_EMBED_MODEL}' not found[/red]")
            console.print(f"Pull it with: ollama pull {OLLAMA_EMBED_MODEL}")
            all_ok = False
    except Exception:
        console.print("[red]ERROR: Cannot connect to Ollama[/red]")
        console.print("Make sure Ollama is installed and running: https://ollama.com/download")
        all_ok = False

    if not all_ok:
        sys.exit(1)

    console.print("[green]✓[/green] NVIDIA API key found")
    console.print("[green]✓[/green] Qdrant is running")
    console.print(f"[green]✓[/green] Ollama model '{OLLAMA_EMBED_MODEL}' is ready")
    console.print()


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Financial Report Analyst — Phase 1 (NVIDIA NIM + Ollama + Qdrant)"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ingest", metavar="PDF_PATH", help="Ingest a PDF document")
    group.add_argument("--ask",    metavar="QUESTION",  help="Ask a question")
    group.add_argument("--list",   action="store_true", help="List all ingested documents")

    parser.add_argument("--company", help="Company name e.g. HDFC, ICICI, SBI")
    parser.add_argument("--quarter", help="Quarter e.g. Q1, Q2, Q3, Q4")
    parser.add_argument("--year",    help="Year e.g. 2024")

    args = parser.parse_args()

    # Run checks before doing anything
    run_startup_checks()

    if args.ingest:
        if not args.company or not args.quarter or not args.year:
            console.print("[red]--ingest requires --company, --quarter and --year[/red]")
            console.print(
                "Example: python financial_analyst.py "
                "--ingest report.pdf --company HDFC --quarter Q3 --year 2024"
            )
            sys.exit(1)
        if not os.path.exists(args.ingest):
            console.print(f"[red]File not found: {args.ingest}[/red]")
            sys.exit(1)
        ingest_document(args.ingest, args.company, args.quarter, args.year)

    elif args.ask:
        ask_question(args.ask, args.company, args.quarter, args.year)

    elif args.list:
        list_documents()


if __name__ == "__main__":
    main()
