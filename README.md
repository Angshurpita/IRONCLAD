# PROJECT IRONCLAD 🛡️

**A minimal, ultra-reliable Retrieval-Augmented Generation (RAG) developer CLI.**

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-58%20passed-brightgreen.svg)]()
[![Code Style](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

> **Philosophy: The Toyota Fortuner of RAG Systems**  
> Proven engineering, relentless reliability, straightforward operation, maintainability, and a narrow set of capabilities executed exceptionally well. Zero fragile agentic loops. Zero proprietary cloud lock-in.

---

## Table of Contents
1. [Overview & Core Architecture](#overview--core-architecture)
2. [Key Capabilities](#key-capabilities)
3. [Prerequisites & Installation](#prerequisites--installation)
4. [Quickstart Guide](#quickstart-guide)
5. [Inference Setup (Google Gemma 4 26B A4B)](#inference-setup)
   - [Option A: Hosted Cloud via OpenRouter](#option-a-hosted-cloud-via-openrouter-recommended)
   - [Option B: Local Offline via Ollama](#option-b-local-offline-via-ollama)
6. [CLI Command Reference](#cli-command-reference)
7. [Technical Guarantees](#technical-guarantees)
   - [Independent Citation Validation](#1-independent-citation-validation)
   - [Strict Abstention on Insufficient Evidence](#2-strict-abstention-on-insufficient-evidence)
   - [Hybrid Search & Reciprocal Rank Fusion (RRF)](#3-hybrid-search--reciprocal-rank-fusion-rrf)
   - [Crash Resilience & Process Concurrency](#4-crash-resilience--process-concurrency)
8. [Configuration & Environment Variables](#configuration--environment-variables)
9. [Troubleshooting & Diagnostics](#troubleshooting--diagnostics)
10. [Development & Testing](#development--testing)

---

## Overview & Core Architecture

PROJECT IRONCLAD is an installable command-line tool built for developers. You point it at any codebase or documentation repository, ingest the contents into a self-contained local knowledge base (`.ironclad/`), and ask natural-language technical questions.

Answers are synthesized using **Google Gemma 4 26B A4B**, strictly grounded on retrieved passages, with every single claim mapped back to exact verified file paths and line ranges.

```
                              ┌───────────────────────────────────┐
                              │     Local Source Code & Docs      │
                              │ (Python, JS, TS, Go, Rust, MD, PDF)│
                              └─────────────────┬─────────────────┘
                                                │
                                       [1. File Discovery]
                                                │ (.gitignore / security filters)
                                                ▼
                                      [2. Parsing & Chunking]
                                                │ (1-indexed line spans, headings)
                                                ▼
                     ┌──────────────────────────┴──────────────────────────┐
                     │                                                     │
                     ▼                                                     ▼
           [3a. Local Embeddings]                                 [3b. Lexical Tokenizer]
           (ONNX / all-MiniLM-L6-v2)                             (Porter Stemmer / Unicode)
                     │                                                     │
                     ▼                                                     ▼
             ChromaDB Vector Store                               SQLite Metadata & FTS5
          (.ironclad/vector_store/)                             (.ironclad/metadata.db)
                     │                                                     │
                     └──────────────────────────┬──────────────────────────┘
                                                │
                                    [4. Hybrid Retrieval]
                                                │ (Dense Vector + BM25 Lexical)
                                                ▼
                                    [5. Rank Fusion (RRF)]
                                                │ (k=60 deduplication & score merging)
                                                ▼
                                   [6. Token Budget Selector]
                                                │ (Hard limit context window packing)
                                                ▼
                                   [7. Google Gemma 4 26B A4B]
                                      OpenRouter (Cloud) / Ollama (Local)
                                                │ (Strict JSON structured generation)
                                                ▼
                                  [8. Citation Verification]
                                                │ (Line span ground-truth validation)
                                                ▼
                                   Grounded Developer Answer
```

---

## Key Capabilities

- **Multi-Format Ingestion**: First-class parsers for source code (Python, JS, TS, Java, Go, Rust, C/C++, Ruby, PHP), Markdown documentation, plain text files, configuration formats, and PDF documents.
- **Strict Grounding & Anti-Hallucination**: Independent post-generation validator checks all model citations against the ground truth lines of the indexed files. Citations referencing non-existent line ranges or hallucinated text are rejected.
- **Disciplined Abstention**: When evidence is missing or ambiguous, Ironclad explicitly abstains rather than inventing plausible-sounding code.
- **Embedded Hybrid Retrieval**:
  - **Dense Vectors**: Persistent ChromaDB vector database using local ONNX embeddings (no external embedding API calls required).
  - **Lexical Search**: SQLite FTS5 with Porter stemming and BM25 ranking for exact keyword, identifier, and symbol matches.
  - **Reciprocal Rank Fusion (RRF)**: Merges dense and sparse result lists with rank normalization.
- **Zero-Downtime Incremental Updates**: Tracks SHA-256 file hashes. Re-ingests only modified or newly created files; purges removed files automatically.
- **Local File Isolation**: All index data lives entirely inside `.ironclad/` within your project root.
- **Privacy Notice**: Explicit console warnings before transmitting source code passages to cloud inference endpoints.

---

## Prerequisites & Installation

### Requirements
- **OS**: macOS (Apple Silicon or Intel), Linux (Ubuntu, Debian, Fedora, Arch), Windows 10/11 (WSL or native PowerShell).
- **Python**: Version 3.11 or newer.
- **SQLite**: Compiled with FTS5 enabled (standard on all modern Python distributions).

### Installation from Source

Clone the repository and install in editable mode:

```bash
git clone https://github.com/ironclad-rag/ironclad.git
cd ironclad

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install dependencies and CLI package
pip install -e .
```

Verify the installation:
```bash
ironclad --version
ironclad doctor
```

---

## Quickstart Guide

### 1. Initialize Ironclad in your project
Navigate to the directory you wish to index and initialize the knowledge base:

```bash
cd /path/to/my-project
ironclad init
```
This generates the `.ironclad/` directory containing SQLite storage and ChromaDB structures.

### 2. Ingest documents and code
Index all source code, Markdown documentation, and PDFs:

```bash
ironclad ingest
```

To view machine-readable JSON metrics of the ingestion run:
```bash
ironclad ingest --json
```

### 3. Ask questions with grounded answers
Ask a technical question answered by Google Gemma:

```bash
ironclad ask "How does user authentication and session validation work?"
```

Output:
```text
╭───────────────────────────── IRONCLAD Answer ──────────────────────────────╮
│ User authentication verifies the incoming credentials against the stored    │
│ hashed password in auth.py [1]. Upon successful verification, a JWT token   │
│ is issued with an expiration time configured by TOKEN_EXPIRY_SECONDS [2].  │
│                                                                            │
│ Model: google/gemma-4-26b-a4b | Latency: 1240ms | Evidence Chunks: 4       │
╰────────────────────────────────────────────────────────────────────────────╯
 Verified Ground-Truth Citations
┏━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ #   ┃ Source File        ┃ Lines     ┃ Excerpt                                         ┃
┡━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ [1] │ src/auth/core.py   │ L42-L58   │ def verify_password(plain, hashed): ...         │
│ [2] │ src/auth/jwt.py    │ L12-L28   │ def issue_session_token(user_id): ...           │
└─────┴────────────────────┴───────────┴─────────────────────────────────────────┘
```

### 4. Search evidence directly (lexical + vector RRF)
Search your codebase without calling an LLM:

```bash
ironclad search "database connection retry logic" --limit 3
```

### 5. Check index status
Inspect total indexed documents, chunk counts, and model consistency:

```bash
ironclad status
```

---

## Inference Setup

PROJECT IRONCLAD is optimized for **Google Gemma 4 26B A4B** (`google/gemma-4-26b-a4b`). You can run it either through cloud inference (OpenRouter) or completely offline (Ollama).

### Option A: Hosted Cloud via OpenRouter (Recommended)

OpenRouter hosts Gemma 4 26B A4B with high throughput and low latency.

1. Obtain an API key from [openrouter.ai](https://openrouter.ai/).
2. Set the key in your shell or create a `.env` file in your project root:

```bash
export OPENROUTER_API_KEY="sk-or-v1-xxxxxxxxxxxxxxxxxxxx"
```

Or configure via `.env`:
```ini
IRONCLAD_INFERENCE_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxx
OPENROUTER_MODEL=google/gemma-4-26b-a4b
```

### Option B: Local Offline via Ollama

To run completely offline with zero data leaving your machine:

1. Install [Ollama](https://ollama.ai/).
2. Pull the Gemma model:
   ```bash
   ollama pull gemma2:27b
   ```
3. Set your environment variables:
   ```bash
   export IRONCLAD_INFERENCE_PROVIDER="ollama"
   export OLLAMA_MODEL="gemma2:27b"
   ```
4. Verify local connectivity:
   ```bash
   ironclad doctor
   ```

---

## CLI Command Reference

| Command | Arguments / Flags | Description |
| :--- | :--- | :--- |
| `ironclad init` | `[PATH]` | Initialize a new `.ironclad/` knowledge base in the target project. |
| `ironclad ingest` | `[PATH]`<br>`--update`<br>`--reindex`<br>`--json` | Scan and index files. `--update` performs incremental delta sync. `--reindex` wipes and rebuilds from scratch. |
| `ironclad ask` | `"<query>"`<br>`-n, --limit INT`<br>`-p, --path PATH`<br>`--json` | Retrieve evidence and synthesize a grounded answer using Gemma with verified citations. |
| `ironclad search` | `"<query>"`<br>`-n, --limit INT`<br>`-p, --path PATH`<br>`--json` | Perform hybrid vector + lexical search and return ranked evidence passages without invoking the LLM. |
| `ironclad status` | `[PATH]`<br>`--json` | Display knowledge base summary, document/chunk counts, and index consistency. |
| `ironclad doctor` | `[PATH]`<br>`--json` | Run diagnostic health checks across Python runtime, storage, SQLite FTS5, embeddings, and inference providers. |
| `ironclad config` | `[PATH]`<br>`--json` | Display active configuration settings with secrets masked. |

---

## Technical Guarantees

### 1. Independent Citation Validation
Unlike naive RAG systems that display whatever references the model produces, Ironclad features an independent citation auditor:
- Evaluates each citation in the LLM output.
- Checks if the cited file path exists in the retrieved evidence list.
- Checks whether the claimed `start_line` and `end_line` fall within the real boundaries of the retrieved passage.
- Rejects citations pointing to non-existent lines or ungrounded sources.
- If the model hallucinated quotes that don't match the evidence, it is flagged as unverified.

### 2. Strict Abstention on Insufficient Evidence
If no relevant documents are retrieved or the retrieved documents do not contain the answer:
- The pipeline immediately abstains without wasting LLM tokens if zero evidence is found.
- If retrieved chunks are insufficient, the system returns an explicit abstention notice:
  > *"I am unable to answer this question because the indexed knowledge base does not contain sufficient evidence."*

### 3. Hybrid Search & Reciprocal Rank Fusion (RRF)
Vector search and lexical search excel at different tasks:
- Dense embeddings capture conceptual semantic similarity.
- Lexical BM25 captures exact identifiers, variable names, and error codes.

Ironclad executes both searches concurrently and normalizes ranks using **Reciprocal Rank Fusion**:
$$RRF(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
where $k = 60$. Documents appearing in both result lists receive a boost, ensuring high precision.

### 4. Crash Resilience & Process Concurrency
- **File Locking**: An atomic PID-based lockfile (`.ironclad/ironclad.lock`) prevents multiple processes from corrupting the index simultaneously.
- **Stale Lock Recovery**: If an ingestion process is killed abruptly (e.g. `SIGKILL` or power failure), subsequent commands inspect the PID via `os.kill(pid, 0)` and clean up stale locks automatically.
- **Transactional SQLite**: Metadata updates and FTS5 synchronization execute within atomic transactions. Ingestion interrupted via `Ctrl+C` cleanly aborts with code `130` without corrupting SQLite or ChromaDB.
- **Embedding Model Mismatch Guard**: If you switch embedding models in configuration, Ironclad flags the inconsistency during `status` and `doctor`, and automatically triggers a safe clean re-index during `ingest` to prevent dimensional mismatch.

---

## Configuration & Environment Variables

All settings can be configured via environment variables or a `.env` file in the project root:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `IRONCLAD_DIR` | `.ironclad` | Subdirectory name for knowledge base storage. |
| `IRONCLAD_INFERENCE_PROVIDER` | `openrouter` | Inference service (`openrouter` or `ollama`). |
| `OPENROUTER_API_KEY` | *(None)* | OpenRouter API authentication key. |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter API base endpoint. |
| `OPENROUTER_MODEL` | `google/gemma-4-26b-a4b` | Hosted Gemma model identifier. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama HTTP service URL. |
| `OLLAMA_MODEL` | `gemma4:26b` | Local Ollama model tag. |
| `IRONCLAD_EMBEDDING_PROVIDER` | `local` | Embedding provider (`local`). |
| `IRONCLAD_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer ONNX embedding model. |
| `IRONCLAD_CHUNK_SIZE` | `500` | Target token size per chunk. |
| `IRONCLAD_CHUNK_OVERLAP` | `50` | Overlap tokens between sequential chunks. |
| `IRONCLAD_MAX_CONTEXT_TOKENS` | `4000` | Hard token budget limit for retrieved evidence. |
| `IRONCLAD_MAX_FILE_SIZE_BYTES` | `10485760` (10 MB) | Maximum individual file size permitted. |
| `IRONCLAD_LOG_LEVEL` | `INFO` | Console log verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |

---

## Troubleshooting & Diagnostics

Run diagnostics at any time to verify system health:
```bash
ironclad doctor
```

### Common Issues & Resolutions

1. **`OPENROUTER_API_KEY is not set`**
   - Solution: Export the key in your terminal (`export OPENROUTER_API_KEY=sk-or-v1-...`) or put it in `.env`.
2. **`Another Ironclad process holds the lock`**
   - Solution: If a previous command died unexpectedly, running `ironclad doctor` or `ironclad ingest` will auto-detect dead PIDs and purge the stale lock. Alternatively, delete `.ironclad/ironclad.lock`.
3. **`Embedding model differs from indexed embedding model`**
   - Solution: Run `ironclad ingest --reindex` to wipe the old dimensional vectors and rebuild with the new model.
4. **`Cannot reach Ollama at http://localhost:11434`**
   - Solution: Ensure Ollama is running (`ollama serve`) and the configured model is installed (`ollama pull gemma4:26b`).

---

## Development & Testing

### Running Tests
Execute the comprehensive test suite (unit and integration tests):

```bash
# Run full test suite
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=ironclad --cov-report=term-missing
```

### Code Formatting & Linting
Ironclad enforces strict typing and linting via Ruff:

```bash
ruff check .
ruff format .
```

---

## License
MIT License. Created for robust, production-minded developer RAG workflows.
