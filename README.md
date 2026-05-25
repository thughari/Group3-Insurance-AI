# 🛡️ Life Insurance AI Copilot

Production-oriented, stateful life-insurance copilot built with **FastAPI + LangGraph + Streamlit**.

## What this app currently does

- Routes each user message to one of 7 specialist flows (`underwriting`, `policy_qa`, `beneficiary`, `issuance`, `lapse_revival`, `policy_comparison`, `lapse_prediction`).
- Applies a 3-tier guardrail pipeline before model execution.
- Uses FAISS-based retrieval over insurance PDFs.
- Uses CSV-based deterministic underwriting helpers for risk and premium indication.
- Persists graph state by session/thread id (memory checkpoint by default, MongoDB-backed active session metadata in API layer).
- Supports synchronous and SSE streaming chat.
- Supports human-in-the-loop pause/resume for higher-risk underwriting outcomes.

---

## Architecture at a glance

- **UI:** `app/ui.py` (Streamlit)
- **API:** `app/main.py` (FastAPI)
- **Workflow graph:** `app/graph.py`
- **State models:** `app/models.py`
- **Guardrails:** `app/guards.py`
- **Caching:** `app/cache.py`
- **Tools:** `app/tools/rag.py`, `app/tools/csv_lookup.py`

---

## Caching strategy (what, where, and why)

The codebase intentionally combines two in-process approaches:

1. **TTLCache (custom, thread-safe)** for data that should expire.
2. **`functools.lru_cache`** for deterministic static file reads.

### Implemented cache layers

From `app/cache.py`:

- `rag_cache = TTLCache(ttl_seconds=600, max_size=128)`
  - Caches RAG retrieval context in `retrieve_policy_context`.
  - Why: repeated queries are common; avoids repeated vector search and formatting cost.

- `guardrail_cache = TTLCache(ttl_seconds=1800, max_size=512)`
  - Caches `apply_guardrails(text)` decisions.
  - Why: guard checks are deterministic for identical input and can include heavier scanner logic.

- `state_cache = TTLCache(ttl_seconds=5, max_size=32)`
  - Short-lived UI state fetch cache.

- `sessions_cache = TTLCache(ttl_seconds=3, max_size=1)`
  - Reduces repeated expensive session-list recalculations.

- `@lru_cache(maxsize=4) read_csv_cached(filepath)`
  - Caches CSV loads used by underwriting tools.
  - Why: static CSV files during runtime; avoids repeated disk I/O.

### LLM response cache

`app/graph.py` configures LangChain global LLM cache:

- `set_llm_cache(InMemoryCache())`

Why this was chosen:
- Zero external infra.
- Very low latency for exact repeated prompts in same process.
- Good fit for single-process/simple deployment.

Tradeoff:
- Process-local and ephemeral (not shared across workers/containers).

### Why not Redis / Memcached (currently)

The code comments explain the decision:
- Keep ops simple (no extra service).
- Avoid network hop for cache access.
- Current implementation is optimized for a single-process style runtime.

When to switch:
- Multi-worker or multi-instance deployments where cache sharing/invalidation consistency is required.

---

## Guardrails strategy (LLM Guard + domain controls)

Guardrails are implemented in `app/guards.py` in three tiers:

1. **Tier 1 (fast domain block list)**
   - Pattern-based insurance/medical safety refusals.
   - Examples: final underwriting decision requests, guaranteed premium claims, medical diagnosis/prescription asks.

2. **Tier 2 (`llm_guard` Anonymize scanner)**
   - Detects sensitive data exposure (PII/PHI-like patterns).

3. **Tier 3 (`llm_guard` PromptInjection scanner)**
   - Detects likely prompt-injection attempts.

Additional behavior:
- Lazy initialization of scanners (`_init_llm_guard`) to reduce startup overhead.
- Graceful fallback to regex-only behavior if scanner init fails.
- Guard results are cached via `guardrail_cache`.

### Why this over regex-only or heavyweight policy engines

- Better balance than regex-only: keeps deterministic business rules + ML-based injection/PII detection.
- Lighter operational complexity than introducing a remote moderation service for every request.
- Keeps critical fail-safe behavior local and predictable.

---

## Intent routing and specialist agents

Implemented in `app/graph.py`:

- `intent_router` (LLM structured output + rule fallback)
- `underwriting_agent`
- `policy_qa_agent`
- `beneficiary_agent`
- `issuance_agent`
- `lapse_revival_agent`
- `policy_comparison_agent`
- `lapse_prediction_agent`
- `human_review`

`underwriting_agent` additionally:
- Extracts applicant fields using structured output.
- Merges with existing applicant state.
- Calls CSV helpers for risk + premium indication.
- Pauses flow for human review when needed.

---

## Retrieval strategy

`app/tools/rag.py`:

- PDF corpus from `app/data/*.pdf`
- Chunking: `chunk_size=1000`, `chunk_overlap=200`
- Vector store: FAISS (local)
- Retrieval: similarity search (`k=3` default)
- Provider marker (`.provider`) triggers index rebuild if embedding provider changes

Embedding provider precedence:
1. GROQ key present → HuggingFace embeddings (`all-MiniLM-L6-v2`)
2. Google/Gemini key present → `gemini-embedding-001`
3. OpenAI key present → OpenAI embeddings

---

## Stateful behavior

- Graph compilation via `build_graph()` in `app/graph.py`.
- Conversation history reducer in `app/models.py` keeps only last 10 messages.
- API layer tracks active sessions and supports session listing/deletion (`app/main.py`).
- Human review is resumed with `/approve`.

---

## API endpoints

From `app/main.py`:

- `GET /health`
- `GET /cache/stats`
- `POST /chat`
- `POST /chat/stream`
- `POST /approve`
- `GET /state/{session_id}`
- `GET /sessions`
- `DELETE /sessions/{session_id}`

---

## Local run

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

In a second terminal:

```bash
streamlit run app/ui.py
```

Or use Docker Compose:

```bash
docker-compose up --build
```

---

## Full technical reference

See `DOCUMENTATION.md` for detailed, file-by-file implementation notes and rationale.
