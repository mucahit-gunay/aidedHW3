"""
retriever.py — Query classification + semantic retrieval from ChromaDB.

Query classification is keyword/rule-based (no extra NLP model):
  - Checks query tokens against known entity names
  - Falls back to signal words ("where", "located" → place; "who", "born" → person)
  - Returns "person", "place", or "both"

Retrieval uses ChromaDB cosine similarity + optional metadata filter.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import Literal, Optional

import chromadb

from embedder import CHROMA_PATH, COLLECTION_NAME, EMBED_MODEL, OLLAMA_URL, get_embedding

log = logging.getLogger("retriever")

QueryType = Literal["person", "place", "both"]

# Signal words for classification
_PERSON_SIGNALS = frozenset(
    "who born died age year career life biography person people "
    "invented discovered wrote painted played sang actor actress "
    "scientist artist musician athlete politician".split()
)
_PLACE_SIGNALS = frozenset(
    "where located built height area country city capital "
    "landmark monument structure tower wall museum temple palace "
    "mountain river lake ocean sea desert forest park island".split()
)

# Known entity names for direct matching (lowercase, split tokens)
from ingest import PEOPLE, PLACES

_ALL_ENTITIES: list[tuple[str, str]] = (
    [(name, "person") for name in PEOPLE]
    + [(name, "place") for name in PLACES]
)

_PERSON_TOKENS: set[str] = set()
for p in PEOPLE:
    _PERSON_TOKENS.update(p.lower().split())

_PLACE_TOKENS: set[str] = set()
for p in PLACES:
    _PLACE_TOKENS.update(p.lower().split())


def detect_entities(query: str) -> list[tuple[str, str]]:
    """Return list of (entity_name, type) explicitly mentioned in the query.

    Uses substring matching for multi-word entity names (e.g. "Eiffel Tower").
    """
    q = query.lower()
    found: list[tuple[str, str]] = []
    for name, etype in _ALL_ENTITIES:
        # Match the full name (case-insensitive) or its most distinctive token.
        name_lower = name.lower()
        if name_lower in q:
            found.append((name, etype))
            continue
        # For single distinctive last names (e.g. "Einstein", "Curie")
        tokens = name_lower.split()
        if len(tokens) >= 2:
            distinctive = tokens[-1]
            if len(distinctive) >= 5 and f" {distinctive}" in f" {q}":
                found.append((name, etype))
    return found

# Common words that appear in both (ignore for classification)
_AMBIGUOUS = frozenset("the of great new".split())


def classify_query(query: str) -> QueryType:
    """
    Determine if query is about a person, place, or both.
    Strategy: count entity name token hits + signal word hits.
    """
    tokens = [t.strip("?.,!").lower() for t in query.split()]

    person_score = 0
    place_score = 0

    for token in tokens:
        if token in _AMBIGUOUS:
            continue
        if token in _PERSON_TOKENS:
            person_score += 2
        if token in _PLACE_TOKENS:
            place_score += 2
        if token in _PERSON_SIGNALS:
            person_score += 1
        if token in _PLACE_SIGNALS:
            place_score += 1

    if person_score > 0 and place_score > 0:
        return "both"
    if person_score > place_score:
        return "person"
    if place_score > person_score:
        return "place"
    return "both"  # default: search everything


def retrieve(
    query: str,
    n_results: int = 5,
    query_type: Optional[QueryType] = None,
) -> list[dict]:
    """
    Embed the query and retrieve top-n similar chunks from ChromaDB.

    Args:
        query: user question
        n_results: number of chunks to return
        query_type: override classification if already known

    Returns:
        list of dicts with keys: text, entity, type, chunk_index, score
    """
    qtype = query_type or classify_query(query)
    mentioned = detect_entities(query)
    log.info("Query type: %s | mentioned entities: %s", qtype, [m[0] for m in mentioned])

    embedding = get_embedding([query], task_prefix="search_query: ")[0]

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # HYBRID RETRIEVAL: if the query mentions specific entities, fetch chunks
    # from those entities first, then top up with semantic search.
    chunks: list[dict] = []
    seen_ids: set[str] = set()

    if mentioned:
        # Fetch top semantic chunks from each mentioned entity
        per_entity = max(2, n_results // len(mentioned))
        for entity_name, _ in mentioned:
            try:
                res = collection.query(
                    query_embeddings=[embedding],
                    n_results=per_entity,
                    where={"entity": {"$eq": entity_name}},
                    include=["documents", "metadatas", "distances"],
                )
            except Exception as e:
                log.error("Entity-filtered query failed for %s: %s", entity_name, e)
                continue
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            dists = res.get("distances", [[]])[0]
            ids = res.get("ids", [[]])[0]
            for doc, meta, dist, _id in zip(docs, metas, dists, ids):
                if _id in seen_ids:
                    continue
                seen_ids.add(_id)
                chunks.append({
                    "text": doc,
                    "entity": meta.get("entity", ""),
                    "type": meta.get("type", ""),
                    "chunk_index": meta.get("chunk_index", 0),
                    "score": round(1 - dist, 4),
                })

    # Top up with general semantic search if we still need more chunks
    remaining = n_results - len(chunks)
    if remaining > 0:
        where_filter: Optional[dict] = None
        if qtype == "person":
            where_filter = {"type": {"$eq": "person"}}
        elif qtype == "place":
            where_filter = {"type": {"$eq": "place"}}

        kwargs: dict = {
            "query_embeddings": [embedding],
            "n_results": n_results + len(seen_ids),  # over-fetch to allow dedup
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            kwargs["where"] = where_filter

        try:
            results = collection.query(**kwargs)
        except Exception as e:
            log.error("ChromaDB query failed: %s", e)
            return chunks

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        for doc, meta, dist, _id in zip(docs, metas, dists, ids):
            if remaining <= 0:
                break
            if _id in seen_ids:
                continue
            seen_ids.add(_id)
            chunks.append({
                "text": doc,
                "entity": meta.get("entity", ""),
                "type": meta.get("type", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "score": round(1 - dist, 4),
            })
            remaining -= 1

    return chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    q = "What did Marie Curie discover?"
    chunks = retrieve(q, n_results=3)
    for c in chunks:
        print(f"[{c['entity']} | {c['type']} | score={c['score']}]")
        print(c["text"][:200])
        print()
