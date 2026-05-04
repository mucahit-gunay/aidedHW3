"""
embedder.py — Local embedding via Ollama + ChromaDB vector store.

Design choice: Option B — one ChromaDB collection with metadata.
  collection name: "wiki_rag"
  metadata per chunk: {type: "person"|"place", entity: "<name>", chunk_index: int}

Why Option B over Option A (two collections)?
  - Comparison queries ("Compare Einstein and Eiffel Tower") need chunks from
    both types in a single retrieval pass.
  - Metadata filtering still lets us scope queries to person/place only.
  - Simpler to maintain one collection vs. two separate index files.

Embedding model: nomic-embed-text via Ollama REST API (localhost:11434).
No external embedding API is used.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import urllib.request
from pathlib import Path
from typing import Optional

import chromadb

log = logging.getLogger("embedder")

DB_PATH = Path(__file__).parent / "data" / "wiki.db"
CHROMA_PATH = Path(__file__).parent / "data" / "chroma"
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
COLLECTION_NAME = "wiki_rag"
BATCH_SIZE = 10  # chunks per Ollama call


def get_embedding(texts: list[str], task_prefix: str = "search_document: ") -> list[list[float]]:
    """Call Ollama embedding API. Returns list of embedding vectors.

    nomic-embed-text requires a task-specific prefix:
      - "search_document: " when embedding chunks for the index
      - "search_query: "    when embedding a user query
    """
    embeddings = []
    for text in texts:
        payload = json.dumps({"model": EMBED_MODEL, "prompt": task_prefix + text}).encode()
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
                embeddings.append(data["embedding"])
        except Exception as e:
            log.error("Embedding failed: %s", e)
            raise
    return embeddings


def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def embed_all(db_path: Path = DB_PATH) -> None:
    """Read all chunks from SQLite, embed via Ollama, store in ChromaDB."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT c.chunk_id, c.entity_id, c.chunk_index, c.text,
               e.name AS entity_name, e.type AS entity_type
        FROM chunks c
        JOIN entities e ON e.entity_id = c.entity_id
    """).fetchall()
    conn.close()

    if not rows:
        log.warning("No chunks found. Run ingest.py and chunker.py first.")
        return

    collection = get_collection()

    # Find already-embedded chunk IDs to skip
    existing = set(collection.get(include=[])["ids"])

    pending = [r for r in rows if str(r["chunk_id"]) not in existing]
    log.info("Total chunks: %d | Already embedded: %d | Pending: %d",
             len(rows), len(existing), len(pending))

    if not pending:
        log.info("All chunks already embedded.")
        return

    # Process in batches
    for i in range(0, len(pending), BATCH_SIZE):
        batch = pending[i: i + BATCH_SIZE]
        texts = [r["text"] for r in batch]
        ids = [str(r["chunk_id"]) for r in batch]
        metadatas = [
            {
                "type": r["entity_type"],
                "entity": r["entity_name"],
                "chunk_index": r["chunk_index"],
                "entity_id": r["entity_id"],
            }
            for r in batch
        ]

        log.info("Embedding batch %d-%d / %d ...", i + 1, i + len(batch), len(pending))
        embeddings = get_embedding(texts)

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        time.sleep(0.1)

    log.info("Embedding complete. %d chunks embedded.", len(pending))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    embed_all()
