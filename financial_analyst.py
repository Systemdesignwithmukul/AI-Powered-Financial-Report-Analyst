"""
Financial Report Analyst — Phase 1 + Phase 2
==============================================
RAG pipeline using:
  - NVIDIA NIM  → LLM (answering questions)    — free API credits
  - Ollama      → Embeddings (nomic-embed-text) — free, runs locally
  - Qdrant      → Vector database               — free, runs in Docker

Phase 1 — Simple RAG:
  python financial_analyst.py --ingest report.pdf --company HDFC --quarter Q3 --year 2024
  python financial_analyst.py --ask "What is the NPA ratio?"
  python financial_analyst.py --list

Phase 2 — ReAct Agent (multi-step reasoning):
  python financial_analyst.py --agent "Compare HDFC and ICICI NPA ratio"
  python financial_analyst.py --agent "How much did HDFC NPA improve from Q2 to Q3?"
  python financial_analyst.py --agent "What companies do you have data for?"
"""

import os
import sys
import json
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

# ── Phase 2 imports (LangChain) ──
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# ─── Setup ────────────────────────────────────────────────────────────────────

load_dotenv()

console = Console()

# ── NVIDIA NIM client ──
NVIDIA_API_KEY   = os.getenv("NVIDIA_API_KEY")
NVIDIA_LLM_MODEL = os.getenv("NVIDIA_LLM_MODEL", "meta/llama-3.1-70b-instruct")

# Direct OpenAI-compatible client → used by Phase 1 generate_answer()
nvidia_client = OpenAI(
    api_key=NVIDIA_API_KEY,
    base_url="https://integrate.api.nvidia.com/v1"
)

# LangChain-compatible client → used by Phase 2 ReAct agent
# Same NVIDIA endpoint, but wrapped in ChatOpenAI so LangChain can use it
langchain_llm = ChatOpenAI(
    model=NVIDIA_LLM_MODEL,
    api_key=NVIDIA_API_KEY,
    base_url="https://integrate.api.nvidia.com/v1",
    temperature=0.1,
    max_tokens=1024
)

# ── Ollama embedding model ──
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# ── Qdrant client ──
qdrant = QdrantClient(host="localhost", port=6333)

COLLECTION_NAME      = "financial_reports"
VECTOR_SIZE          = 768   # nomic-embed-text output size
AGENT_MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "10"))


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

            text = page.extract_text()
            if text:
                page_content += text

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

            if len(chunk_text.strip()) > 50:
                chunks.append({
                    "id":   f"chunk_{chunk_id}",
                    "text": chunk_text,
                    "page": page_num
                })
                chunk_id += 1

            start += chunk_size - overlap

    console.print(
        f"[green]✓[/green] Created {len(chunks)} chunks "
        f"(chunk_size={chunk_size} words, overlap={overlap} words)"
    )
    return chunks


# ─── STEP 3: Embedding (Ollama) ───────────────────────────────────────────────

def get_embedding(text: str) -> list[float]:
    """
    Converts text into a 768-dimensional vector using Ollama's nomic-embed-text.
    Runs locally — no API call, no cost, no internet needed.
    """
    response = ollama.embeddings(
        model=OLLAMA_EMBED_MODEL,
        prompt=text
    )
    return response["embedding"]


def embed_in_batches(texts: list[str], batch_size: int = 50) -> list[list[float]]:
    """Embeds a large list of texts and logs progress every 50 chunks."""
    all_embeddings = []

    for i, text in enumerate(texts):
        all_embeddings.append(get_embedding(text))
        if (i + 1) % 50 == 0:
            console.print(f"  Embedded {i + 1}/{len(texts)} chunks...")

    console.print(f"  Embedded {len(texts)}/{len(texts)} chunks...")
    return all_embeddings


# ─── STEP 4: Store in Qdrant ──────────────────────────────────────────────────

def store_in_qdrant(chunks, embeddings, company, quarter, year, pdf_path):
    """
    Stores chunk text + embedding vectors in Qdrant.
    Uses upsert — safe to re-run without duplicating data.
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

    Score meaning:
      1.0 = perfect match  |  0.8+ = very relevant  |  < 0.3 = not useful
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


# ─── STEP 6: Answer Generation — Phase 1 (NVIDIA NIM direct) ─────────────────

def generate_answer(query: str, context_chunks: list[dict]) -> str:
    """
    Phase 1 answer generation.
    Sends question + retrieved chunks directly to NVIDIA NIM.
    temperature=0.1 → factual, consistent answers.
    """
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


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — ReAct Agent with LangChain Tools
# ═══════════════════════════════════════════════════════════════════════════════
#
# What changed from Phase 1:
#   Phase 1: you call retrieve_relevant_chunks() + generate_answer() manually
#   Phase 2: a ReAct agent decides which tool to call, calls it, reads
#            the result, thinks again, calls another tool if needed,
#            and produces a final answer — all automatically
#
# The @tool decorator converts a plain Python function into a LangChain tool.
# The agent reads the docstring to decide WHEN to use each tool.
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def search_financial_reports(query: str) -> str:
    """
    Search for information in ingested financial reports using semantic search.

    Use this tool when you need to find specific financial data, metrics,
    ratios, or any information from the financial documents.

    Input: a natural language search query string.
    Example inputs:
      - "HDFC Bank NPA ratio Q3 2024"
      - "ICICI capital adequacy ratio"
      - "net interest margin quarterly results"

    Returns: relevant text chunks from documents with source citations.
    """
    try:
        query_embedding = get_embedding(query)

        results = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=4,
            with_payload=True
        )

        if not results:
            return "No relevant information found for this query."

        output = f"Search results for '{query}':\n\n"
        for i, hit in enumerate(results):
            p = hit.payload
            output += (
                f"[Result {i + 1}] "
                f"{p.get('company')} · {p.get('quarter')} {p.get('year')} "
                f"· Page {p.get('page')} "
                f"(relevance: {round(hit.score, 2)})\n"
                f"{p.get('text', '')}\n\n"
            )

        return output

    except Exception as e:
        return f"Search error: {str(e)}"


@tool
def compare_companies(input_json: str) -> str:
    """
    Compare a specific financial metric across two or more companies.

    Use this tool when asked to compare companies on any metric
    like NPA ratio, NIM, ROE, capital adequacy, loan growth etc.

    Input must be a JSON string with these fields:
      - metric: the metric to compare (e.g. "NPA ratio")
      - companies: list of company names (e.g. ["HDFC", "ICICI"])
      - quarter: quarter (e.g. "Q3")
      - year: year (e.g. "2024")

    Example input:
      {"metric": "gross NPA ratio", "companies": ["HDFC", "ICICI"], "quarter": "Q3", "year": "2024"}

    Returns: search results for each company to help you compare.
    """
    try:
        data      = json.loads(input_json)
        metric    = data.get("metric", "")
        companies = data.get("companies", [])
        quarter   = data.get("quarter", "")
        year      = data.get("year", "")

        output = f"Comparison of '{metric}' across companies:\n\n"

        for company in companies:
            query           = f"{company} {metric} {quarter} {year}"
            query_embedding = get_embedding(query)

            results = qdrant.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_embedding,
                limit=2,
                query_filter=Filter(
                    must=[FieldCondition(
                        key="company",
                        match=MatchValue(value=company.upper())
                    )]
                ),
                with_payload=True
            )

            output += f"--- {company.upper()} ---\n"
            if results:
                for hit in results:
                    p = hit.payload
                    output += (
                        f"[{p.get('quarter')} {p.get('year')} · Page {p.get('page')}]\n"
                        f"{p.get('text', '')}\n\n"
                    )
            else:
                output += f"No data found for {company}. Make sure you ingested {company}'s report.\n\n"

        return output

    except json.JSONDecodeError:
        return (
            "Invalid JSON. Use format:\n"
            '{"metric": "NPA ratio", "companies": ["HDFC", "ICICI"], "quarter": "Q3", "year": "2024"}'
        )
    except Exception as e:
        return f"Comparison error: {str(e)}"


@tool
def calculate_ratio(input_json: str) -> str:
    """
    Calculate financial ratios or simple financial calculations.

    Use this tool when you need to:
      - Calculate percentage change between two values
      - Find difference between two financial metrics
      - Calculate growth rates

    Input must be a JSON string with these fields:
      - calculation_type: "percentage_change", "difference", or "growth_rate"
      - value1: first number (current value)
      - value2: second number (previous/base value)
      - label: description of what you're calculating

    Example input:
      {"calculation_type": "percentage_change", "value1": 1.26, "value2": 1.34, "label": "HDFC NPA Q3 vs Q2"}

    Returns: calculated result with explanation.
    """
    try:
        data      = json.loads(input_json)
        calc_type = data.get("calculation_type", "percentage_change")
        value1    = float(data.get("value1", 0))
        value2    = float(data.get("value2", 0))
        label     = data.get("label", "")

        if calc_type == "percentage_change":
            if value2 == 0:
                return "Cannot calculate — base value is 0."
            change    = ((value1 - value2) / value2) * 100
            direction = "increased" if change > 0 else "decreased"
            return (
                f"Calculation: {label}\n"
                f"Value 1 : {value1}\n"
                f"Value 2 : {value2}\n"
                f"Result  : {direction} by {abs(round(change, 2))}%"
            )

        elif calc_type == "difference":
            diff      = value1 - value2
            direction = "higher" if diff > 0 else "lower"
            return (
                f"Calculation: {label}\n"
                f"Value 1 : {value1}\n"
                f"Value 2 : {value2}\n"
                f"Result  : {abs(round(diff, 4))} {direction}"
            )

        elif calc_type == "growth_rate":
            if value2 == 0:
                return "Cannot calculate — base value is 0."
            rate = ((value1 - value2) / value2) * 100
            return (
                f"Calculation: {label}\n"
                f"Current  : {value1}\n"
                f"Previous : {value2}\n"
                f"Growth   : {round(rate, 2)}%"
            )

        return f"Unknown calculation_type: {calc_type}"

    except json.JSONDecodeError:
        return "Invalid JSON input for calculation."
    except Exception as e:
        return f"Calculation error: {str(e)}"


@tool
def list_available_documents(query: str = "") -> str:
    """
    List all financial documents currently available in the database.

    Use this tool when:
      - You need to know which companies are available
      - You need to check which quarters or years are ingested
      - The user asks what documents are available
      - Before comparing companies, verify both are ingested

    Input: any string (not used — just pass empty string "")

    Returns: list of all ingested documents with chunk counts.
    """
    try:
        results, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=10000,
            with_payload=True,
            with_vectors=False
        )

        if not results:
            return "No documents ingested yet. Please ingest financial reports first."

        docs = {}
        for point in results:
            p   = point.payload
            key = f"{p.get('company')} · {p.get('quarter')} {p.get('year')}"
            docs[key] = docs.get(key, 0) + 1

        output = "Available documents in the database:\n\n"
        for doc, count in sorted(docs.items()):
            output += f"  {doc}  ({count} chunks)\n"

        return output

    except Exception as e:
        return f"Could not list documents: {str(e)}"


# ─── PHASE 2: Build and Run ReAct Agent ───────────────────────────────────────

def build_agent() -> AgentExecutor:
    """
    Builds a tool-calling agent — works better with Llama 3.1
    than ReAct because Llama supports native function/tool calling.

    ReAct requires strict text formatting (Thought/Action/Observation)
    which Llama often gets wrong.

    Tool calling uses structured JSON under the hood — much more reliable.
    """
    tools = [
        search_financial_reports,
        compare_companies,
        calculate_ratio,
        list_available_documents,
    ]

    # Tool calling prompt — simpler than ReAct, no strict format needed
    prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert financial analyst AI assistant.
You have access to financial reports in a vector database.

You MUST follow these rules on EVERY single question — no exceptions:

RULE 1 — ALWAYS call list_available_documents first.
RULE 2 — ALWAYS call search_financial_reports at least TWICE
         with different search terms before answering.
RULE 3 — First search: use the exact term from the question.
         Second search: use alternate financial terms:
           NPA       → try "gross NPA non performing assets asset quality"
           Capital   → try "CRAR CAR Basel tier 1 capital ratio"
           Income    → try "NII net interest income net profit PAT"
           Margin    → try "NIM net interest margin yield on advances"
           Revenue   → try "total income operating revenue fee income"
RULE 4 — After both searches, combine results and give Final Answer.
RULE 5 — NEVER give Final Answer after just 1 tool call.
RULE 6 — Always cite: company · quarter · year · page number.
RULE 7 — Never say data is unavailable without doing 2 searches first.

You MUST use minimum 3 tool calls before giving any Final Answer."""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

    agent = create_tool_calling_agent(
        llm=langchain_llm,
        tools=tools,
        prompt=prompt
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=AGENT_MAX_ITERATIONS,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )


def ask_agent(query: str):
    console.print(Panel(
        f"[bold]Agent Question:[/bold] {query}\n"
        f"[dim]Mode: Phase 2 Tool Calling Agent[/dim]",
        style="cyan"
    ))

    agent_executor = build_agent()

    try:
        result = agent_executor.invoke({"input": query})

        # Manually print each step cleanly — shows which tools were called
        steps = result.get("intermediate_steps", [])
        if steps:
            console.print(f"\n[cyan]Agent took {len(steps)} step(s):[/cyan]")
            for i, (action, observation) in enumerate(steps):
                console.print(f"\n  [bold cyan]Step {i + 1}[/bold cyan]")
                console.print(f"  Tool   : [yellow]{action.tool}[/yellow]")
                console.print(f"  Input  : {action.tool_input}")
                console.print(f"  Result : {str(observation)[:200]}...")
        else:
            console.print("\n[cyan]Agent answered directly (no tools needed)[/cyan]")

        # Print final answer exactly once
        console.print(Panel(
            Markdown(result["output"]),
            title="[bold green]Final Answer[/bold green]",
            style="green"
        ))

    except Exception as e:
        console.print(f"[red]Agent error: {str(e)}[/red]")


# ─── PHASE 1: ask_question (unchanged) ───────────────────────────────────────

def ask_question(query: str, company: str = None, quarter: str = None, year: str = None):
    """
    Phase 1 — simple RAG pipeline.
    Question → embed → retrieve → NVIDIA NIM → answer
    """
    console.print(Panel(
        f"[bold]Question:[/bold] {query}\n"
        f"[dim]Mode: Phase 1 Simple RAG[/dim]",
        style="cyan"
    ))

    filters = {}
    if company: filters["company"] = company.upper()
    if quarter: filters["quarter"] = quarter.upper()
    if year:    filters["year"]    = str(year)

    console.print("\n[cyan]Searching Qdrant for relevant chunks...[/cyan]")
    chunks = retrieve_relevant_chunks(
        query,
        n_results=5,
        filters=filters if filters else None
    )

    if not chunks:
        console.print("[red]No relevant chunks found.[/red]")
        console.print(
            "  python financial_analyst.py "
            "--ingest report.pdf --company HDFC --quarter Q3 --year 2024"
        )
        return

    console.print(f"\n[cyan]Top {len(chunks)} relevant chunks found:[/cyan]")
    for i, chunk in enumerate(chunks):
        meta = chunk["metadata"]
        console.print(
            f"  [{i + 1}] {meta.get('company')} · {meta.get('quarter')} "
            f"{meta.get('year')} · Page {meta.get('page')} "
            f"(score: {chunk['score']})"
        )

    console.print(f"\n[cyan]Generating answer with {NVIDIA_LLM_MODEL} (NVIDIA NIM)...[/cyan]")
    answer = generate_answer(query, chunks)

    console.print(Panel(
        Markdown(answer),
        title="[bold green]Answer (Phase 1 RAG)[/bold green]",
        style="green"
    ))


# ─── Ingestion + List (unchanged from Phase 1) ────────────────────────────────

def ingest_document(pdf_path: str, company: str, quarter: str, year: str):
    console.print(Panel(
        f"[bold]Ingesting document[/bold]\n\n"
        f"File    : {pdf_path}\n"
        f"Company : {company.upper()}\n"
        f"Quarter : {quarter.upper()}\n"
        f"Year    : {year}\n\n"
        f"Embedding : {OLLAMA_EMBED_MODEL} (Ollama)\n"
        f"LLM       : {NVIDIA_LLM_MODEL} (NVIDIA NIM)",
        style="cyan"
    ))

    ensure_collection_exists()

    pages  = extract_text_from_pdf(pdf_path)
    chunks = split_into_chunks(pages, chunk_size=500, overlap=100)

    console.print(f"\n[cyan]Generating embeddings with Ollama ({OLLAMA_EMBED_MODEL})...[/cyan]")
    texts      = [c["text"] for c in chunks]
    embeddings = embed_in_batches(texts)
    console.print(f"[green]✓[/green] Embeddings done")

    store_in_qdrant(chunks, embeddings, company, quarter, year, pdf_path)

    console.print(Panel(
        f"[bold green]Ingestion complete![/bold green]\n\n"
        f"Stored {len(chunks)} chunks for {company.upper()} {quarter.upper()} {year}\n\n"
        f"Phase 1 — simple question:\n"
        f'  python financial_analyst.py --ask "What is the NPA ratio?"\n\n'
        f"Phase 2 — agent reasoning:\n"
        f'  python financial_analyst.py --agent "Compare HDFC and ICICI NPA ratio"',
        style="green"
    ))


def list_documents():
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
    all_ok = True

    if not NVIDIA_API_KEY:
        console.print("[red]ERROR: NVIDIA_API_KEY not found in .env file[/red]")
        console.print("Get your free key from: https://build.nvidia.com")
        all_ok = False

    try:
        qdrant.get_collections()
    except Exception:
        console.print("[red]ERROR: Cannot connect to Qdrant[/red]")
        console.print("Run: docker start qdrant")
        all_ok = False

    try:
        available = [m["name"] for m in ollama.list()["models"]]
        base      = OLLAMA_EMBED_MODEL.split(":")[0]
        if not any(base in m for m in available):
            console.print(f"[red]ERROR: Ollama model '{OLLAMA_EMBED_MODEL}' not found[/red]")
            console.print(f"Run: ollama pull {OLLAMA_EMBED_MODEL}")
            all_ok = False
    except Exception:
        console.print("[red]ERROR: Ollama not running[/red]")
        console.print("Install from: https://ollama.com/download")
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
        description="Financial Report Analyst — Phase 1 + Phase 2"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ingest", metavar="PDF_PATH", help="Ingest a PDF document")
    group.add_argument("--ask",    metavar="QUESTION",  help="Phase 1: simple RAG question")
    group.add_argument("--agent",  metavar="QUESTION",  help="Phase 2: ReAct agent question")
    group.add_argument("--list",   action="store_true", help="List all ingested documents")

    parser.add_argument("--company", help="Company name e.g. HDFC, ICICI, SBI")
    parser.add_argument("--quarter", help="Quarter e.g. Q1, Q2, Q3, Q4")
    parser.add_argument("--year",    help="Year e.g. 2024")

    args = parser.parse_args()

    run_startup_checks()

    if args.ingest:
        if not args.company or not args.quarter or not args.year:
            console.print("[red]--ingest requires --company, --quarter and --year[/red]")
            sys.exit(1)
        if not os.path.exists(args.ingest):
            console.print(f"[red]File not found: {args.ingest}[/red]")
            sys.exit(1)
        ingest_document(args.ingest, args.company, args.quarter, args.year)

    elif args.ask:
        # Phase 1 — simple RAG
        ask_question(args.ask, args.company, args.quarter, args.year)

    elif args.agent:
        # Phase 2 — ReAct agent (no company/quarter filter needed — agent decides)
        ask_agent(args.agent)

    elif args.list:
        list_documents()


if __name__ == "__main__":
    main()
