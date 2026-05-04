"""Quick smoke test: classify, retrieve, generate for sample queries."""

from retriever import classify_query, retrieve
from generator import generate

TESTS = [
    ("Who was Albert Einstein and what is he known for?", "person"),
    ("What did Marie Curie discover?", "person"),
    ("Where is the Eiffel Tower located?", "place"),
    ("What was the Colosseum used for?", "place"),
    ("Compare Lionel Messi and Cristiano Ronaldo", "person"),
    ("Which person is associated with electricity?", "person"),
    ("Who is the president of Mars?", "both"),
]


def main():
    for q, expected_type in TESTS:
        qtype = classify_query(q)
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print(f"Classified: {qtype}  (expected hint: {expected_type})")
        chunks = retrieve(q, n_results=4)
        print(f"Retrieved {len(chunks)} chunks. Top entities: {[c['entity'] for c in chunks]}")
        ans = generate(q, chunks)
        print(f"A: {ans[:300]}")


if __name__ == "__main__":
    main()
