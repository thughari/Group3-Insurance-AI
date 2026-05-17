---
title: Life Insurance AI Copilot
emoji: 🏢
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# 🛡️ Life Insurance AI Copilot

Production-grade **LangGraph stateful** life insurance copilot built for scalability and performance. 

### Key Features
- **7 Specialist Agent Nodes** + 2 bonus innovation agents via dynamic Intent Routing.
- **MongoDB Persistent State:** Deep session persistence across server restarts using `langgraph-checkpoint-mongodb` and `pymongo`.
- **Session Management:** Dedicated Streamlit sidebar to seamlessly switch between active sessions, view conversation intents/traces, and securely delete sessions.
- **Human-in-the-Loop (HitL):** Pauses graph execution for human underwriter review on high-risk profiles.
- **High-Performance Streaming:** True real-time token streaming via SSE (`astream_events`) integrated flawlessly with Streamlit UI.
- **In-Memory LLM Caching:** Intercepts duplicate exact queries globally using LangChain's `InMemoryCache` for 0ms latency and cost savings.
- **Context Window Optimization:** Custom LangGraph reducer automatically truncates the conversation memory to the last 10 messages to protect LLM context limits.
- **🎙️ Multimodal Input:** Voice & image upload support (OpenAI Whisper + GPT-4o-mini).
- **RAG & Structured Lookups:** FAISS-powered PDF retrieval + Pandas CSV actuarial lookups.

---

## Architecture

### System Architecture & Data Flow (5 Layers)

```mermaid
flowchart LR
    subgraph Layer1["1. User / UI"]
        UI["Streamlit Chat\n+ Multimodal Input\n+ Session Switching"]
    end

    subgraph Layer2["2. API Layer"]
        API["FastAPI Service\n/chat /chat/stream\n/approve /state /sessions"]
    end

    subgraph Layer3["3. LangGraph Workflow"]
        LG["Stateful Agents\nIntent Router → Specialists\nHitL Interrupt"]
    end

    subgraph Layer4["4. Persistence & Knowledge"]
        KB["FAISS Vector Store\n+ CSV Lookups"]
        DB[(MongoDB Atlas\nState Checkpoints\n+ Active Sessions)]
    end

    subgraph Layer5["5. Cloud Infra"]
        CLOUD["Docker Container\nGCP Cloud Run"]
    end

    UI -->|"1. Query / Switch Session"| API
    API -->|"2. Graph execution"| LG
    LG -->|"3. RAG / CSV"| KB
    LG -->|"4. Checkpoint / Update"| DB
    API -.->|"SSE Token stream\n(0-latency UI)"| UI
    CLOUD ---|"hosts"| API

    style Layer1 fill:#2d6a4f,stroke:#1b4332,color:#fff
    style Layer2 fill:#264653,stroke:#1d3557,color:#fff
    style Layer3 fill:#6a4c93,stroke:#4a2d73,color:#fff
    style Layer4 fill:#e76f51,stroke:#c44536,color:#fff
    style Layer5 fill:#457b9d,stroke:#1d3557,color:#fff
```

**Figure 1:** System architecture showcasing MongoDB integration and streaming UI.

---

### LangGraph Stateful Workflow

```mermaid
flowchart TD
    START(("🟢 START")) --> IR

    IR["🔷 Intent Router\nClassifies query type"]

    IR -->|underwriting| UW["🔷 Underwriting Agent\nRisk intake & scoring"]
    IR -->|policy_qa| PQ["🔷 Policy Q&A Agent\nRAG over policy docs"]
    IR -->|beneficiary| BN["🔷 Beneficiary Agent\nNomination & queries"]
    IR -->|issuance| IS["🔷 Issuance Agent\nPolicy issue guidance"]
    IR -->|lapse_revival| LR["🔷 Lapse & Revival Agent\nGrace period & reinstatement"]
    IR -->|policy_comparison| PC["⭐ Policy Comparison Agent\nStructured comparison tables"]
    IR -->|lapse_prediction| LP["⭐ Lapse Prediction Agent\nPayment history risk analysis"]

    UW -->|"high risk?"| HR["🔴 Human Review\nUnderwriter approval - HitL"]
    UW -->|"standard risk"| SS
    HR -->|"approved / rejected"| SS

    PQ --> SS
    BN --> SS
    IS --> SS
    LR --> SS
    PC --> SS
    LP --> SS

    SS[("🟣 MongoDB Shared State\napplicant_data · risk_tier\nTruncated history (max 10)\nnode_path · node_outputs")]

    SS --> END1(("⚫ END"))
```

---

## Performance & Optimization Additions
- **Concurrent API Fetching:** The `/sessions` endpoint uses `asyncio.gather` to concurrently query MongoDB for all active session states, preventing UI lockups and ensuring instant Streamlit sidebar re-renders.
- **LLM Token Conservation:** LangChain's `InMemoryCache` prevents the system from making expensive duplicate LLM API calls.
- **Dynamic Context Window:** The `conversation_history` typed dictionary uses a custom Python reducer (`add_and_truncate_history`) to guarantee the array never exceeds 10 items, preventing context window bloat during long sessions.
- **Native Async Streaming:** The backend executes `llm.astream_events` entirely asynchronously, ensuring the FastAPI event loop is never blocked during token generation.

---

## Quickstart

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd Life-Insurance-AI
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory:
```env
# Choose one of the following LLM providers:
OPENAI_API_KEY=your-openai-key
# GOOGLE_API_KEY=your-gemini-key
# GROQ_API_KEY=your-groq-key

# Persistent State Database (Required for session continuity):
MONGODB_URI=mongodb+srv://<user>:<password>@cluster.mongodb.net/

# LangSmith Tracing (required for observability):
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-key
LANGCHAIN_PROJECT=life-insurance-copilot
```

### 3. Run the Application

You can run the application either using Docker (recommended) or locally using Python.

#### Option A: Using Docker
Start the services using Docker Compose:
```bash
docker-compose up --build
```
Access the Streamlit UI at `http://localhost:8501`.

#### Option B: Local Setup (Two Terminals)
**Terminal 1 (Backend):**
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 (Frontend):**
```bash
.\venv\Scripts\activate
python -m streamlit run app/ui.py
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness check |
| `/chat` | POST | Synchronous chat execution |
| `/chat/stream` | POST | **Streaming chat (SSE) - Powers the UI** |
| `/approve` | POST | Submit HitL underwriter approval to resume graph |
| `/state/{session_id}` | GET | Retrieve full LangGraph state checkpoint |
| `/sessions` | GET | **Concurrent fetch of all active sessions** |
| `/sessions/{session_id}` | DELETE | Securely delete a session from DB tracking |

---

## Repo Layout

```text
├── app/
│   ├── main.py            # FastAPI endpoints + SSE Streaming + Async Session Management
│   ├── graph.py           # LangGraph workflow + InMemory LLM Caching
│   ├── models.py          # TypedDict state + Custom Context Truncator
│   ├── guards.py          # Safety guardrails
│   ├── ui.py              # Streamlit chat UI + Sidebar Session Tracking
│   ├── tools/
│   │   ├── csv_lookup.py  # Actuarial Risk classification from CSV
│   │   └── rag.py         # FAISS vector store build + retrieval
│   └── data/              # PDFs + CSVs + FAISS index
├── evaluation/
│   └── run_eval.py        # DeepEval scorecard
├── Dockerfile             # Containerized for GCP Cloud Run
├── docker-compose.yml
└── requirements.txt       # Heavily optimized for production
```
