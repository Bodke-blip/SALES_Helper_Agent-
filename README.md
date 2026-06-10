# Predikly Sales Helper Agent

An internal agentic AI sales assistant for finding and reusing Predikly customer use cases and case-study knowledge. The app lets a user ask sales questions, retrieve grounded internal context from Qdrant, and generate concise answers or sales drafts through a controlled LangGraph workflow.

## What It Does

- Answers questions about previous Predikly customer use cases.
- Retrieves internal case-study context from a Qdrant vector database.
- Supports count-style queries such as how many use cases exist for a customer.
- Drafts short sales content using retrieved internal context.
- Applies input guardrails for prompt injection, unsafe requests, and credential extraction.
- Evaluates retrieved context before returning a final response.
- Returns source metadata, trace IDs, cache status, model details, and workflow timings.

## Architecture

The runtime flow is:

```text
User / Web UI / API
  -> FastAPI app
  -> LangGraph workflow
  -> Input guardrail
  -> Main orchestrator
  -> Knowledge retrieval agent
  -> Evaluation agent
  -> Response composer or fallback handler
  -> Output guardrail
  -> Final response
```

The ingestion flow is:

```text
Google Drive + reference Excel/PPT files
  -> metadata extraction
  -> dense embeddings with sentence-transformers/all-MiniLM-L6-v2
  -> BM25 sparse vectors
  -> Qdrant hybrid collection
  -> customer manifest
```

## Tech Stack

- Python
- FastAPI
- LangGraph
- LangChain
- Qdrant
- Google Gemini
- Hosted Hugging Face MiniLM embeddings
- Optional Langfuse tracing

## Project Structure

```text
.
├── app.py                         # FastAPI app, UI, and API endpoints
├── requirements.txt               # Python dependencies
├── agents/
│   ├── graph.py                   # LangGraph workflow
│   ├── orchestrator_agent.py      # Intent routing and final response composition
│   ├── knowledge_retrieval_agent.py
│   ├── eval_agent.py
│   ├── evaluation.py
│   ├── guardrails.py
│   ├── fallback.py
│   ├── llm.py
│   ├── state.py
│   └── tracing.py
└── data/                          # Local generated data, ignored by Git
```

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```bash
touch .env
```

Add the required configuration:

```env
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
HYBRID_QDRANT_COLLECTION_NAME=predikly_hybrid_search_data_v2
HYBRID_QDRANT_FALLBACK_COLLECTION_NAME=predikly_hybrid_serch_data

GEMINI_API_KEY=your_gemini_api_key
PRIMARY_LLM_TIMEOUT_SECONDS=12

HF_TOKEN=your_huggingface_token
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_TIMEOUT_SECONDS=20

ENABLE_LANGFUSE_TRACING=false
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=
```

Optional retrieval settings:

```env
QDRANT_TOP_K=15
QDRANT_HYBRID_PREFETCH_LIMIT=50
QDRANT_DOCUMENT_EXPANSION_SOURCE_LIMIT=3
QDRANT_DOCUMENT_EXPANSION_CHUNK_LIMIT=40
QDRANT_MAX_CONTEXT_ITEMS=30
QDRANT_TIMEOUT_SECONDS=8
ENSURE_PAYLOAD_INDEXES_ON_QUERY=false
MIN_RETRIEVAL_RESULTS=1
QDRANT_UPSERT_BATCH_SIZE=50
BM25_STATE_PATH=data/bm25_sparse_encoder.json
RETRIEVAL_CACHE_TTL_SECONDS=300
RETRIEVAL_CACHE_MAX_ENTRIES=256
MAX_CHAT_HISTORY_TURNS=6
CHAT_HISTORY_LIMIT=40
CHAT_DB_URL=postgresql://USER:PASSWORD@HOST:5432/predikly_sales_helper?sslmode=require
```

## Secrets and Local Files

Do not commit local secrets or OAuth files. The `.gitignore` excludes:

- `.env`
- `google_token.json`
- `client_secret_*.json`
- `.venv/`
- `__pycache__/`
- `data/customer_manifest.json`

For Google Drive ingestion, keep the OAuth client secret JSON locally in the project root and set `GOOGLE_CLIENT_SECRET_FILE` to its filename. The first ingestion run may create `google_token.json`.

## Run the App

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

Open the web UI:

```text
http://127.0.0.1:8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## Deploy on Render

This repo includes a `render.yaml` blueprint and Dockerfile for a Render web
service. Render builds the container from the repo and serves the FastAPI app on
the `PORT` environment variable.

Before deploying, make sure `data/bm25_sparse_encoder.json` is committed. The
rest of `data/` stays ignored.

In Render, create a new Blueprint or Web Service from the GitHub repo and set
these secret environment variables:

```env
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
GEMINI_API_KEY=your_gemini_api_key
HF_HUB_TOKEN=your_huggingface_token
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
```

The blueprint sets the non-secret defaults, including:

```env
PRIMARY_LLM_TIMEOUT_SECONDS=45
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
BM25_STATE_PATH=data/bm25_sparse_encoder.json
ENABLE_LANGFUSE_TRACING=true
LANGFUSE_HOST=https://cloud.langfuse.com
```

After deploy, test:

```bash
curl https://YOUR-RENDER-URL.onrender.com/health
```

## Query API

Example request:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "List the relevant case studies for a healthcare client.",
    "use_gemini_llm": true,
    "use_local_llm": false,
    "verbose": true
  }'
```

Useful options:

- `use_gemini_llm`: enables Gemini.
- `use_local_llm`: accepted for backward compatibility, but local LLM fallback is disabled in this lightweight runtime.
- `verbose`: includes agent trace details.

## Cache Endpoints

Check retrieval cache:

```bash
curl http://127.0.0.1:8000/cache/status
```

Clear retrieval cache:

```bash
curl -X DELETE http://127.0.0.1:8000/cache
```

## Ingestion and Qdrant Upload

Ingestion, upload, and evaluation dataset tooling is local-only and intentionally excluded from this public repository because it can reference internal customer material. The hosted app expects Qdrant collections and sparse encoder state to be prepared through the private/local ingestion flow.

## Notes

- The default primary model is `gemini-2.5-flash`.
- Query embeddings are generated through the hosted Hugging Face endpoint for `sentence-transformers/all-MiniLM-L6-v2`.
- The default hybrid Qdrant collection is `predikly_hybrid_search_data_v2`.
- The default fallback hybrid Qdrant collection is `predikly_hybrid_serch_data`.
- Retrieval checks the main hybrid collection first and only tries the fallback collection when the main collection returns fewer than `MIN_RETRIEVAL_RESULTS`.
- This project is currently shaped for internal development/pilot use. Add authentication, production secret management, monitoring, CI checks, and a formal security review before production deployment.
