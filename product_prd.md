# Product Requirements Document — Wikipedia RAG Assistant

## Overview

A locally-hosted retrieval-augmented generation (RAG) system that answers natural language questions about famous people and places using Wikipedia as its knowledge base.

## Problem Statement

General-purpose LLMs hallucinate facts and require internet connectivity. A local RAG system grounds answers in verified Wikipedia content and runs entirely offline after initial data ingestion.

## Goals

- Answer factual questions about 40+ famous people and places with high accuracy
- Run entirely on localhost with no external API calls during inference
- Provide a simple, intuitive chat interface
- Return "I don't know" gracefully when data is unavailable

## Non-Goals

- Real-time Wikipedia updates
- Multi-language support
- User authentication

## System Components

### Ingest
- Source: Wikipedia REST API (`/w/api.php?action=query&prop=extracts&explaintext=true`)
- Entities: 20 famous people, 20 famous places
- Storage: SQLite (`data/wiki.db`) — tables: `entities`, `chunks`
- Politeness: 0.3s delay between requests, proper User-Agent header

### Chunking
- Strategy: sliding window, 200 words per chunk, 40-word overlap
- Rationale: overlap preserves sentence context at chunk boundaries; 200 words fits Ollama's embedding context window comfortably

### Embedding & Vector Store
- Model: `nomic-embed-text` via Ollama (local, no API key)
- Vector DB: ChromaDB (persistent, cosine similarity)
- Design: single collection `wiki_rag` with metadata (`type`, `entity`, `chunk_index`)
- Batch size: 10 chunks per Ollama call to avoid memory spikes

### Query Classification
- Method: keyword/rule-based (no extra model)
- Person signals: "who", "born", "discovered", "invented", known person name tokens
- Place signals: "where", "located", "built", "height", known place name tokens
- Fallback: "both" — search entire collection without filter

### Retrieval
- Hybrid: entity-name detection in query → entity-filtered semantic search per mentioned entity
- Fallback: cosine similarity over the entire collection (or scoped by `type` metadata)
- De-duplicated, top-N chunks returned
- Rationale: pure cosine search underranked the actually-named entity; lexical pre-filter dramatically improves recall for direct questions and comparison queries

### Generation
- Model: `llama3.2:3b` via Ollama
- Temperature: 0.1 (factual, low creativity)
- System prompt instructs model to answer only from context
- Streaming enabled for responsive UI

### UI
- Framework: Streamlit
- Features: chat history, source chunk viewer, query type indicator, example buttons, clear button

## Data

### Required Entities

**People:** Albert Einstein, Marie Curie, Leonardo da Vinci, William Shakespeare, Ada Lovelace, Nikola Tesla, Lionel Messi, Cristiano Ronaldo, Taylor Swift, Frida Kahlo, Isaac Newton, Charles Darwin, Mahatma Gandhi, Nelson Mandela, Cleopatra, Napoleon Bonaparte, Abraham Lincoln, Wolfgang Amadeus Mozart, Vincent van Gogh, Stephen Hawking

**Places:** Eiffel Tower, Great Wall of China, Taj Mahal, Grand Canyon, Machu Picchu, Colosseum, Hagia Sophia, Statue of Liberty, Pyramids of Giza, Mount Everest, Stonehenge, Acropolis of Athens, Angkor Wat, Chichen Itza, Petra, Sagrada Familia, Sydney Opera House, Burj Khalifa, Amazon rainforest, Niagara Falls

## Success Criteria

- All 40 required entities successfully ingested and indexed
- Factual questions answered correctly from context
- "I don't know" returned for out-of-scope questions
- System starts and responds within 30 seconds on a standard laptop
