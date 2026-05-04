"""
generator.py — Answer generation via local Ollama LLM.

Takes a user query + retrieved context chunks, builds a prompt,
and streams the response from Ollama (llama3.2:3b).
Returns "I don't know" when context is empty or model signals no answer.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import Generator, Optional

log = logging.getLogger("generator")

OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "llama3.2:3b"

SYSTEM_PROMPT = """You are a Wikipedia-grounded assistant specialized in famous people and places.

Follow these rules strictly:
- Direct questions (who, what, where, when, why): give one direct sentence answer first, then supporting details from context.
- Comparison questions: use "Similarities" and "Differences" sections, drawing only from the provided sources.
- Partial knowledge: answer what the context supports, then explicitly note "The available data does not cover [missing aspect]."
- No relevant context at all: respond exactly with "I don't know based on the available data."
- Never invent or infer numbers, dates, statistics, or quotes not present in the context.
- When stating a specific fact, cite its source in parentheses: (Source 1), (Source 2), etc."""


def build_prompt(query: str, chunks: list[dict]) -> str:
    if not chunks:
        context = "No relevant information found."
    else:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(f"[Source {i}: {chunk['entity']} ({chunk['type']})]\n{chunk['text']}")
        context = "\n\n".join(parts)

    return f"""Context:
{context}

Question: {query}

Answer:"""


def generate(query: str, chunks: list[dict], stream: bool = False) -> str:
    """
    Generate an answer using Ollama LLM.
    Returns the full answer string.
    """
    if not chunks:
        return "I don't know based on the available data."

    prompt = build_prompt(query, chunks)

    payload = json.dumps({
        "model": LLM_MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 512,
        },
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
            answer = data.get("response", "").strip()
            if not answer:
                return "I don't know based on the available data."
            return answer
    except Exception as e:
        log.error("LLM generation failed: %s", e)
        return f"Error: could not generate answer ({e})"


def generate_stream(query: str, chunks: list[dict]) -> Generator[str, None, None]:
    """
    Stream answer tokens from Ollama. Yields string fragments.
    """
    if not chunks:
        yield "I don't know based on the available data."
        return

    prompt = build_prompt(query, chunks)

    payload = json.dumps({
        "model": LLM_MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.1,
            "num_predict": 512,
        },
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for line in resp:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    token = data.get("response", "")
                    if token:
                        yield token
                    if data.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        log.error("Stream generation failed: %s", e)
        yield f"\n[Error: {e}]"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from retriever import retrieve
    q = "What did Marie Curie discover?"
    chunks = retrieve(q, n_results=4)
    print("Answer:", generate(q, chunks))
