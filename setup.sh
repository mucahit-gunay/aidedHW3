#!/bin/bash
# One-shot setup: ingest → chunk → embed
set -e
echo "=== Step 1: Ingesting Wikipedia data ==="
python ingest.py

echo "=== Step 2: Chunking documents ==="
python chunker.py

echo "=== Step 3: Embedding and indexing ==="
python embedder.py

echo "=== Done! Run: streamlit run app.py ==="
