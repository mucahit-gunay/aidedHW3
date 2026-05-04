# Wikipedia RAG Assistant

A local, ChatGPT-style question-answering system about famous people and places. Built on Wikipedia data, ChromaDB vector search, and Ollama local LLMs. Runs entirely on localhost — no external APIs.

---

## Architecture

```
Wikipedia API
     ↓ ingest.py
SQLite (entities + raw text)
     ↓ chunker.py
SQLite (chunks)
     ↓ embedder.py (nomic-embed-text via Ollama)
ChromaDB (vectors + metadata)
     ↓ retriever.py (query classifier + cosine search)
     ↓ generator.py (llama3.2:3b via Ollama)
Streamlit UI (app.py)
```

**Design choice — Option B (single collection):** One ChromaDB collection with `type` metadata (`person` / `place`). This allows comparison queries across both types while still supporting filtered retrieval when the query is clearly about one type.

**Chunking strategy:** Sliding window, 200 words per chunk, 40-word overlap. Large documents are safely handled in O(n) time without full string copies.

**Hybrid retrieval:** Pure cosine search on `nomic-embed-text` was found to underrank chunks of the actually-named entity (e.g. "Where is the Eiffel Tower?" matched generic "X is located in Y" chunks from Petra, Machu Picchu, etc. higher than the Eiffel Tower chunk itself). The retriever now first detects whether the query mentions any known entity name (substring + last-name match) and pulls top semantic chunks from those entities directly, then tops up with general semantic search. This combines lexical precision with embedding recall.

**Embedding prefixes:** `nomic-embed-text` requires task-specific prefixes — `search_document: ` for indexed chunks and `search_query: ` for queries. Without them, retrieval quality drops noticeably.

---

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running

---

## Setup

### 1. Install dependencies

```bash
cd hw3
pip install -r requirements.txt
```

### 2. Pull local models via Ollama

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Make sure Ollama is running in the background:

```bash
ollama serve
```

---

## Running the System

### Step 1 — Ingest Wikipedia data

```bash
python ingest.py
```

Fetches 40 Wikipedia pages (20 people + 20 places) and stores plain text in `data/wiki.db`.

### Step 2 — Chunk the documents

```bash
python chunker.py
```

Splits each article into overlapping 200-word chunks and stores them in SQLite.

### Step 3 — Embed and index

```bash
python embedder.py
```

Generates embeddings for all chunks using `nomic-embed-text` and stores them in ChromaDB at `data/chroma/`.

### Step 4 — Start the chat application

```bash
streamlit run app.py
```

Opens the chat UI at [http://localhost:8501](http://localhost:8501).

### Optional — Smoke test the pipeline

```bash
python test_pipeline.py
```

Runs 7 sample queries end-to-end (classify → retrieve → generate) and prints results.

---

## Example Queries

**People**
- Who was Albert Einstein and what is he known for?
- What did Marie Curie discover?
- Why is Nikola Tesla famous?
- Compare Lionel Messi and Cristiano Ronaldo

**Places**
- Where is the Eiffel Tower located?
- Why is the Great Wall of China important?
- What was the Colosseum used for?
- Where is Mount Everest?

**Mixed**
- Which famous place is located in Turkey?
- Which person is associated with electricity?
- Compare the Eiffel Tower and the Statue of Liberty

**Failure cases (expected "I don't know")**
- Who is the president of Mars?
- Tell me about John Doe

---

## Project Structure

```
hw3/
├── app.py              # Streamlit chat UI
├── ingest.py           # Wikipedia → SQLite ingestion
├── chunker.py          # Text chunking
├── embedder.py         # Ollama embeddings → ChromaDB
├── retriever.py        # Query classification + semantic search
├── generator.py        # Ollama LLM answer generation
├── crawler/
│   ├── utils.py        # HTML parsing, tokenization (from hw2)
│   └── storage.py      # SQLite helpers (from hw2)
├── data/               # Created at runtime
│   ├── wiki.db         # SQLite database
│   └── chroma/         # ChromaDB vector store
├── requirements.txt
├── README.md
├── product_prd.md
└── recommendation.md
```
# aidedHW3
