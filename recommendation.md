# Production Deployment Recommendation

## Current Architecture (Local / Prototype)

| Component | Local Choice | Limitation |
|---|---|---|
| LLM | Ollama llama3.2:3b | Low throughput, single user |
| Embeddings | Ollama nomic-embed-text | Sequential, slow at scale |
| Vector DB | ChromaDB (file-based) | No multi-node, no replication |
| Data store | SQLite | No concurrent writes at scale |
| UI | Streamlit | Not production-grade |

---

## Recommended Production Stack

### LLM
Replace Ollama with a managed inference endpoint:
- **Option A (privacy-first):** Self-hosted vLLM cluster with `llama3.1-8b` or `mistral-7b` behind a load balancer. Enables batching, GPU acceleration, and horizontal scaling.
- **Option B (cost-efficient):** Anthropic Claude API or Together AI for inference — fast, no GPU management required.

### Embeddings
- Replace sequential Ollama embedding with **batch inference** using `sentence-transformers` on GPU, or use a managed service like Cohere Embed.
- Cache embeddings in the vector DB — never re-embed unchanged documents.

### Vector Database
Replace ChromaDB with a production vector store:
- **Qdrant** or **Weaviate** — both support multi-node, filtering, and REST APIs
- **pgvector** — if already using PostgreSQL, adds vector search with no extra infra

### Relational Database
Replace SQLite with **PostgreSQL**:
- Concurrent writes from multiple ingestion workers
- Full-text search complement to vector search (hybrid retrieval)

### Ingestion Pipeline
- Schedule Wikipedia re-ingestion weekly via a job scheduler (Celery, Airflow)
- Add change detection: re-embed only updated articles

### API Layer
- Wrap retrieval + generation behind a **FastAPI** service
- Add request queuing and rate limiting
- Cache recent query/answer pairs in Redis to reduce LLM calls

### UI
Replace Streamlit with a proper frontend (React / Next.js) backed by the FastAPI service.

### Monitoring
- Track retrieval quality (mean cosine similarity per query)
- Track LLM latency, token usage
- Alert on "I don't know" rate spikes (signals data gaps)

---

## Tradeoffs Summary

| Concern | Local Prototype | Production |
|---|---|---|
| Cost | Free | GPU/API costs |
| Privacy | Full (no external calls) | Partial (managed services) |
| Latency | 5–30s | < 2s with GPU + caching |
| Scalability | Single user | Thousands of concurrent users |
| Maintenance | Manual | CI/CD, monitoring required |

## Minimum Viable Production Path

1. Move SQLite → PostgreSQL + pgvector (drop-in replacement, low migration cost)
2. Deploy Ollama on a GPU machine (A10 or better) behind an nginx reverse proxy
3. Add Redis caching for repeated queries
4. Containerize with Docker Compose

This path preserves full data privacy (no external APIs) while supporting 10–50 concurrent users with acceptable latency.
