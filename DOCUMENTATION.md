# Life Insurance AI Copilot — Implementation Documentation

This document reflects the **current codebase** and explains implementation choices, especially caching, guardrails, and workflow decisions.

## 1) Codebase map

- `app/main.py` — FastAPI app, endpoints, session tracking, streaming transport.
- `app/graph.py` — LangGraph nodes, routing, LLM provider selection, graph build/checkpointer.
- `app/models.py` — request/response schemas + graph state definition + reducers.
- `app/guards.py` — 3-tier guardrail pipeline with `llm_guard` scanners.
- `app/cache.py` — centralized in-memory caching primitives and instances.
- `app/tools/rag.py` — PDF ingestion, FAISS indexing, retrieval, provider compatibility marker.
- `app/tools/csv_lookup.py` — deterministic risk classification + indicative premium lookup.
- `app/ui.py` — Streamlit frontend.

## 2) Caching: options considered and why this design won

### Implemented now

The app uses **in-process cache layers** from `app/cache.py`:

- `TTLCache` (custom): expiring key-value cache with thread lock.
- `lru_cache` for static CSV reads.
- LangChain `InMemoryCache` for repeated LLM calls (`app/graph.py`).

### Why this was chosen

1. **Zero extra infrastructure**
   - No Redis/Memcached service needed.
2. **Low latency**
   - Local memory lookup is faster than network round-trip.
3. **Simplicity + maintainability**
   - Custom TTL implementation is compact and easy to audit.
4. **Fit for current runtime shape**
   - Single-process friendly behavior is sufficient for local/dev/small deployments.

### Alternatives considered and why not chosen (for now)

- **Redis / Memcached**
  - Pros: shared cache across workers and instances.
  - Cons: infra overhead, deployment complexity, network latency.
  - Decision: postpone until horizontal scaling requires cross-instance cache coherence.

- **Third-party Python cache libraries (e.g., cachetools)**
  - Pros: battle-tested utilities.
  - Cons: extra dependency for a small feature that stdlib + tiny custom code already covers.
  - Decision: keep dependency footprint smaller.

### Where each cache is used

- `rag_cache` (TTL 600s): caches FAISS retrieval output in `retrieve_policy_context`.
- `guardrail_cache` (TTL 1800s): caches `apply_guardrails` results.
- `state_cache` (TTL 5s): short-lived UI state support.
- `sessions_cache` (TTL 3s): `/sessions`-style list optimization.
- `read_csv_cached` (`lru_cache(maxsize=4)`): caching for underwriting CSV file reads.
- LangChain `InMemoryCache`: repeated identical LLM calls in graph nodes.

### Tradeoffs and future path

- **Current tradeoff:** process-local only, not shared across workers.
- **Future path:** swap TTL caches and LLM cache backend to Redis when deploying multiple worker processes/replicas.

## 3) Guardrails: options considered and final architecture

### Implemented now (3 tiers)

In `app/guards.py`, the pipeline is:

1. **Tier 1 — Domain regex/keyword policy**
   - Fast, deterministic policy boundaries.
2. **Tier 2 — `llm_guard` Anonymize scanner**
   - Sensitive-data detection.
3. **Tier 3 — `llm_guard` PromptInjection scanner**
   - Injection-risk detection.

### Why this was chosen over alternatives

- **Over regex-only:** catches more subtle injection/sensitive-input patterns than handcrafted rules alone.
- **Over pure ML moderation-only:** preserves strict, explainable insurance-domain rules with deterministic refusals.
- **Over remote moderation dependency:** keeps core safety checks local, reducing external coupling for this stage.

### Operational behavior

- Scanners are lazily initialized once (`_init_llm_guard`).
- If scanner setup fails, system degrades gracefully to Tier-1 regex behavior (no hard crash).
- Results are cached to avoid repeated scanner cost for duplicate input.

## 4) LLM/provider strategy

### Chat model selection (`app/graph.py`)

Provider precedence:
1. `GROQ_API_KEY` → `llama-3.3-70b-versatile`
2. `GOOGLE_API_KEY` / `GEMINI_API_KEY` → `gemini-2.5-flash`
3. `OPENAI_API_KEY` → `gpt-4o-mini`

Rationale:
- Environment-driven provider flexibility.
- Single code path for multi-provider fallback.

### Embedding selection (`app/tools/rag.py`)

Provider precedence:
1. GROQ key present → HuggingFace `all-MiniLM-L6-v2`
2. Google/Gemini key present → `gemini-embedding-001`
3. OpenAI key present → OpenAI embeddings

Rationale:
- Keeps embeddings available under whichever provider keys are configured.

## 5) Retrieval architecture

- Source: PDFs in `app/data/`
- Loader: `PyPDFLoader`
- Splitter: `RecursiveCharacterTextSplitter(1000, 200)`
- Index: local FAISS directory (`app/data/faiss_index`)
- Search: similarity search (`k=3` default)
- Compatibility guard: `.provider` marker triggers rebuild if embedding provider changes

Why FAISS:
- Local, fast vector search with no managed vector DB dependency.
- Suitable for contained document corpus and easy local deployment.

## 6) Underwriting logic

In `app/tools/csv_lookup.py`:

- `classify_risk(disclosures)`
  - Deterministic mapping from disclosures to project tiers.
  - Uses direct smoker heuristic + CSV condition matching.

- `indicative_premium_lookup(age, cover_amount, term_years, risk_tier)`
  - Nearest-neighbor lookups on age/term/cover.
  - Risk-tier-driven row selection strategy.
  - Formula fallback when rows missing.

Why deterministic CSV logic:
- Transparent, auditable, and reproducible for baseline underwriting guidance.
- Prevents opaque model-only pricing behavior.

## 7) Stateful conversation and history management

In `app/models.py`:

- `CopilotState` typed state for graph.
- `conversation_history` reducer truncates to last 10 entries (`add_and_truncate_history`).

Why:
- Controls context growth and token cost.
- Maintains short but relevant interaction memory.

## 8) Human-in-the-loop design

In graph/API coordination:
- High-risk/substandard underwriting can set `requires_human_review` and pause at `human_review` node.
- `/approve` endpoint writes human decision then resumes graph execution.

Why:
- Keeps final sensitive outcomes under explicit human oversight.

## 9) API surface

- `GET /health`
- `GET /cache/stats`
- `POST /chat`
- `POST /chat/stream`
- `POST /approve`
- `GET /state/{session_id}`
- `GET /sessions`
- `DELETE /sessions/{session_id}`

## 10) Note on legacy typo file

A legacy file `DDOCUMENATION.md` exists in the repo. `DOCUMENTATION.md` is the updated canonical version.
