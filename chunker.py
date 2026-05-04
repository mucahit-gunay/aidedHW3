"""
chunker.py — Sliding-window text chunker.

Splits a document into overlapping chunks by word count.
No external libraries — uses only str.split().

Strategy: fixed-size words with overlap.
  chunk_size=200 words, overlap=40 words.
  Designed for large documents: O(n) time, no full-copy on each slice.
"""

from __future__ import annotations

import sqlite3
import logging
from pathlib import Path

log = logging.getLogger("chunker")

CHUNK_SIZE = 200   # words per chunk
OVERLAP = 40       # words shared between consecutive chunks
DB_PATH = Path(__file__).parent / "data" / "wiki.db"


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """
    Split text into overlapping word-based chunks.
    Returns list of chunk strings.
    """
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    step = chunk_size - overlap
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step

    return chunks


def chunk_all(db_path: Path = DB_PATH) -> None:
    """Read all entities from SQLite, chunk their text, write chunks back."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")

    entities = conn.execute(
        "SELECT entity_id, name, raw_text FROM entities WHERE raw_text IS NOT NULL"
    ).fetchall()

    total_chunks = 0
    for entity in entities:
        existing = conn.execute(
            "SELECT COUNT(*) as c FROM chunks WHERE entity_id=?",
            (entity["entity_id"],)
        ).fetchone()
        if existing and existing["c"] > 0:
            log.info("Already chunked: %s", entity["name"])
            continue

        chunks = chunk_text(entity["raw_text"])
        rows = [
            (entity["entity_id"], idx, chunk)
            for idx, chunk in enumerate(chunks)
        ]
        conn.executemany(
            "INSERT OR IGNORE INTO chunks (entity_id, chunk_index, text) VALUES (?,?,?)",
            rows,
        )
        conn.commit()
        total_chunks += len(chunks)
        log.info("Chunked '%s': %d chunks", entity["name"], len(chunks))

    conn.close()
    log.info("Chunking complete. %d new chunks total.", total_chunks)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    chunk_all()
