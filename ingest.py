"""
ingest.py — Wikipedia data ingestion into SQLite.

Fetches Wikipedia page content via the public REST API (no external library),
parses plain text using hw2's utils, and stores in a local SQLite database.
Designed around the hw2 Storage schema with two extra tables: entities, chunks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional
import urllib.request
import urllib.parse
import urllib.error

from crawler.utils import extract_text

log = logging.getLogger("ingest")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

DB_PATH = Path(__file__).parent / "data" / "wiki.db"

PEOPLE: list[str] = [
    "Albert Einstein", "Marie Curie", "Leonardo da Vinci",
    "William Shakespeare", "Ada Lovelace", "Nikola Tesla",
    "Lionel Messi", "Cristiano Ronaldo", "Taylor Swift", "Frida Kahlo",
    "Isaac Newton", "Charles Darwin", "Mahatma Gandhi", "Nelson Mandela",
    "Cleopatra", "Napoleon Bonaparte", "Abraham Lincoln", "Wolfgang Amadeus Mozart",
    "Vincent van Gogh", "Stephen Hawking",
]

PLACES: list[str] = [
    "Eiffel Tower", "Great Wall of China", "Taj Mahal", "Grand Canyon",
    "Machu Picchu", "Colosseum", "Hagia Sophia", "Statue of Liberty",
    "Pyramids of Giza", "Mount Everest",
    "Stonehenge", "Acropolis of Athens", "Angkor Wat", "Chichen Itza",
    "Petra", "Sagrada Familia", "Sydney Opera House", "Burj Khalifa",
    "Amazon rainforest", "Niagara Falls",
]


def init_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS entities (
            entity_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL UNIQUE,
            type        TEXT NOT NULL,  -- 'person' or 'place'
            wiki_url    TEXT,
            raw_text    TEXT,
            fetched_at  REAL
        );
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id   INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            text        TEXT NOT NULL,
            UNIQUE(entity_id, chunk_index),
            FOREIGN KEY(entity_id) REFERENCES entities(entity_id)
        );
    """)
    conn.commit()
    return conn


def fetch_wikipedia(title: str, max_retries: int = 5) -> Optional[str]:
    """Fetch plain text of a Wikipedia article via the REST API.

    Retries on HTTP 429 (rate limit) with exponential backoff.
    """
    encoded = urllib.parse.quote(title.replace(" ", "_"))
    url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=true&titles={encoded}&format=json&redirects=1"
    headers = {
        "User-Agent": "hw3-rag-edu/1.0 (mucahitgunay@itu.edu.tr)",
        "Accept": "application/json",
    }

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                text = page.get("extract", "")
                if text and len(text) > 100:
                    return text
            return None
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 2 ** attempt + 2  # 3, 4, 6, 10, 18 seconds
                log.warning("Rate limited on '%s', waiting %ds (attempt %d/%d)",
                            title, wait, attempt + 1, max_retries)
                time.sleep(wait)
                continue
            log.warning("HTTP %d for '%s'", e.code, title)
            return None
        except Exception as e:
            log.warning("Failed to fetch '%s': %s", title, e)
            time.sleep(2)
    log.error("Gave up on '%s' after %d retries", title, max_retries)
    return None


def ingest_entity(conn: sqlite3.Connection, name: str, entity_type: str) -> bool:
    existing = conn.execute(
        "SELECT entity_id FROM entities WHERE name=?", (name,)
    ).fetchone()
    if existing:
        log.info("Already ingested: %s", name)
        return False

    log.info("Fetching: %s", name)
    text = fetch_wikipedia(name)
    if not text:
        log.warning("No content for: %s", name)
        return False

    encoded = urllib.parse.quote(name.replace(" ", "_"))
    wiki_url = f"https://en.wikipedia.org/wiki/{encoded}"

    conn.execute(
        "INSERT INTO entities (name, type, wiki_url, raw_text, fetched_at) VALUES (?,?,?,?,?)",
        (name, entity_type, wiki_url, text, time.time()),
    )
    conn.commit()
    log.info("Saved: %s (%d chars)", name, len(text))
    return True


def ingest_all(db_path: Path = DB_PATH) -> None:
    conn = init_db(db_path)
    total = 0
    for name in PEOPLE:
        if ingest_entity(conn, name, "person"):
            total += 1
        time.sleep(1.5)  # polite delay (Wikipedia is strict)

    for name in PLACES:
        if ingest_entity(conn, name, "place"):
            total += 1
        time.sleep(1.5)

    conn.close()
    log.info("Ingestion complete. %d new entities.", total)


if __name__ == "__main__":
    ingest_all()
