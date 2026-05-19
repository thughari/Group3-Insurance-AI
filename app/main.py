import json
import asyncio
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.graph import build_graph, stream_agent_response
from app.guards import apply_guardrails
from app.models import ChatRequest, ChatResponse, CopilotState

app = FastAPI(title="Life Insurance AI Copilot")
compiled_graph = build_graph()

import os
import pymongo
import certifi

SESSIONS_FILE = os.path.join(os.path.dirname(__file__), "data", "sessions.json")
mongodb_uri = os.getenv("MONGODB_URI")

mongo_client = None
mongo_collection = None
if mongodb_uri:
    mongo_client = pymongo.MongoClient(mongodb_uri, tlsCAFile=certifi.where())
    mongo_collection = mongo_client["copilot_db"]["active_sessions"]

def load_sessions():
    if mongo_collection is not None:
        sessions = {}
        for doc in mongo_collection.find():
            sessions[doc["_id"]] = doc["data"]
        return sessions
    else:
        if os.path.exists(SESSIONS_FILE):
            try:
                with open(SESSIONS_FILE, "r") as f:
                    return json.load(f)
            except:
                return {}
        return {}

def save_sessions(sessions):
    if mongo_collection is not None:
        for sid, data in sessions.items():
            mongo_collection.update_one({"_id": sid}, {"$set": {"data": data}}, upsert=True)
    else:
        os.makedirs(os.path.dirname(SESSIONS_FILE), exist_ok=True)
        with open(SESSIONS_FILE, "w") as f:
            json.dump(sessions, f)

# Track active sessions for the /sessions endpoint
_active_sessions: dict[str, dict] = load_sessions()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    guard = apply_guardrails(req.message)
    if guard.blocked:
        return ChatResponse(
            session_id=req.session_id,
            node_path=["guardrail_block"],
            response=guard.reason,
            state={"guarded": True},
        )

    config = {"configurable": {"thread_id": req.session_id}}

    import datetime
    _active_sessions[req.session_id] = {
        "last_active": datetime.datetime.now().isoformat(),
        "last_query": req.message[:100],
    }
    save_sessions(_active_sessions)

    # Check if currently interrupted
    state_snapshot = await compiled_graph.aget_state(config)
    if state_snapshot and state_snapshot.next:
        if "human_review" in state_snapshot.next:
            return ChatResponse(
                session_id=req.session_id,
                node_path=["human_review_pending"],
                response="Your application is currently paused pending human underwriter review. Please wait for an underwriter to approve or reject it.",
                state=dict(state_snapshot.values) if state_snapshot.values else {},
            )

    # Invoke the LangGraph workflow
    result = await compiled_graph.ainvoke(
        {"user_query": req.message, "session_id": req.session_id},
        config=config,
    )

    if not isinstance(result, dict):
        if hasattr(result, "model_dump"):
            result = result.model_dump()
        else:
            result = dict(result)

    # Check if interrupted after invoke
    state_snapshot = await compiled_graph.aget_state(config)
    is_paused = bool(state_snapshot and state_snapshot.next)

    response_text = result.get("response", "")
    if is_paused and "human_review" in (state_snapshot.next or []):
        response_text += "\n\n⚠️ **[SYSTEM: High-risk/substandard case. PAUSING for Human Underwriter review. Please approve or reject in the sidebar.]**"

    # Update conversation history in state (appended via reducer)
    await compiled_graph.aupdate_state(config, {
        "conversation_history": [
            {"role": "user", "content": req.message},
            {"role": "assistant", "content": response_text},
        ]
    })

    return ChatResponse(
        session_id=req.session_id,
        node_path=result.get("node_path", []),
        response=response_text,
        state=result,
    )


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE streaming endpoint — streams LLM response tokens in real time."""

    guard = apply_guardrails(req.message)
    if guard.blocked:
        async def blocked_stream():
            yield f"data: {json.dumps({'type': 'blocked', 'content': guard.reason})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(blocked_stream(), media_type="text/event-stream")

    config = {"configurable": {"thread_id": req.session_id}}

    # Track session
    import datetime
    _active_sessions[req.session_id] = {
        "last_active": datetime.datetime.now().isoformat(),
        "last_query": req.message[:100],
    }
    save_sessions(_active_sessions)

    # Check if currently interrupted
    state_snapshot = await compiled_graph.aget_state(config)
    if state_snapshot and state_snapshot.next:
        if "human_review" in state_snapshot.next:
            async def paused_stream():
                yield f"data: {json.dumps({'type': 'paused', 'content': 'Your application is currently paused pending human underwriter review.'})}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(paused_stream(), media_type="text/event-stream")

    async def event_stream():
        response_text = ""
        node_path = []
        meta_sent = False
        is_paused = False

        try:
            async for event in compiled_graph.astream_events(
                {"user_query": req.message, "session_id": req.session_id},
                config,
                version="v2",
            ):
                kind = event["event"]
                
                if kind == "on_chat_model_stream" and "final_response" in event.get("tags", []):
                    if not meta_sent:
                        # We don't have the final node_path yet, but we can send an empty or partial one
                        yield f"data: {json.dumps({'type': 'meta', 'node_path': [], 'is_paused': False})}\n\n"
                        meta_sent = True
                        
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, "content") and chunk.content:
                        response_text += chunk.content
                        yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"
        except Exception as e:
            error_msg = f"\n\nError generating response: {str(e)}"
            yield f"data: {json.dumps({'type': 'token', 'content': error_msg})}\n\n"

        # After execution completes (or pauses), check the final state
        state_snapshot = await compiled_graph.aget_state(config)
        is_paused = bool(state_snapshot and state_snapshot.next)
        
        if not meta_sent:
            if state_snapshot and state_snapshot.values:
                node_path = state_snapshot.values.get("node_path", [])
            yield f"data: {json.dumps({'type': 'meta', 'node_path': node_path, 'is_paused': is_paused})}\n\n"

        # Fallback for cached responses:
        # If the LLM response was cached, `astream_events` does not emit `on_chat_model_stream` chunks.
        # As a result, `response_text` will be empty. We can retrieve the full response from the final state!
        if not response_text and state_snapshot and state_snapshot.values:
            cached_response = state_snapshot.values.get("response", "")
            if cached_response:
                response_text = cached_response
                yield f"data: {json.dumps({'type': 'token', 'content': cached_response})}\n\n"

        if is_paused:
            pause_msg = "\n\n⚠️ **[SYSTEM: High-risk/substandard case. PAUSING for Human Underwriter review.]**"
            yield f"data: {json.dumps({'type': 'token', 'content': pause_msg})}\n\n"
            response_with_pause = response_text + pause_msg
        else:
            response_with_pause = response_text

        # Persist conversation history after streaming
        await compiled_graph.aupdate_state(config, {
            "conversation_history": [
                {"role": "user", "content": req.message},
                {"role": "assistant", "content": response_with_pause if is_paused else response_text},
            ]
        })

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


class ApprovalRequest(BaseModel):
    session_id: str
    approved: bool


@app.post("/approve")
async def approve(req: ApprovalRequest):
    config = {"configurable": {"thread_id": req.session_id}}
    state_snapshot = await compiled_graph.aget_state(config)

    if not state_snapshot or "human_review" not in (state_snapshot.next or []):
        raise HTTPException(status_code=400, detail="No pending human review found for this session.")

    # Update state with approval and resume
    await compiled_graph.aupdate_state(config, {
        "approved_by_human": req.approved,
        "user_query": "Human Review Completed.",
        "response": f"Underwriter decision: {'✅ Approved' if req.approved else '❌ Rejected'}.",
    })

    # Resume execution
    result = await compiled_graph.ainvoke(None, config=config)

    if hasattr(result, "model_dump"):
        result = result.model_dump()
    elif not isinstance(result, dict):
        result = dict(result)

    return {"status": "resumed", "state": result}


@app.get("/state/{session_id}")
async def get_state(session_id: str):
    snapshot = await compiled_graph.aget_state({"configurable": {"thread_id": session_id}})
    if not snapshot:
        return {"session_id": session_id, "state": {}}

    is_paused = bool(snapshot.next and "human_review" in snapshot.next)

    if hasattr(snapshot.values, "model_dump"):
        data = snapshot.values.model_dump()
    else:
        data = dict(snapshot.values) if snapshot.values else {}

    data["is_paused"] = is_paused
    return {"session_id": session_id, "state": data}


@app.get("/sessions")
async def list_sessions():
    """List all active sessions with their last activity."""
    sessions = []
    
    try:
        if mongo_collection is not None:
            current_sessions = load_sessions()
            global _active_sessions
            _active_sessions = current_sessions
        else:
            current_sessions = _active_sessions
            
        import asyncio

        async def fetch_session_info(sid, info):
            try:
                snapshot = await compiled_graph.aget_state({"configurable": {"thread_id": sid}})
                is_paused = bool(snapshot and snapshot.next and "human_review" in snapshot.next)
                node_path = []
                intent = "unknown"
                if snapshot and snapshot.values:
                    vals = dict(snapshot.values) if not hasattr(snapshot.values, "model_dump") else snapshot.values.model_dump()
                    node_path = vals.get("node_path", [])
                    intent = vals.get("intent", "unknown")
                
                # Robust parsing for legacy sessions
                last_active = info.get("last_active", "N/A") if isinstance(info, dict) else str(info)
                last_query = info.get("last_query", "N/A") if isinstance(info, dict) else "N/A"
                
                return {
                    "session_id": sid,
                    "last_active": last_active,
                    "last_query": last_query,
                    "is_paused": is_paused,
                    "intent": intent,
                    "node_path": node_path,
                }
            except Exception as e:
                print(f"Error fetching state for {sid}: {e}")
                return None

        tasks = [fetch_session_info(sid, info) for sid, info in current_sessions.items()]
        results = await asyncio.gather(*tasks)
        sessions = [s for s in results if s is not None]
        
        # Sort by last_active descending so newest is at the top
        sessions.sort(key=lambda x: x["last_active"], reverse=True)
        
        return {"count": len(sessions), "sessions": sessions}
    except Exception as e:
        print(f"Critical error in /sessions: {e}")
        # Return empty list gracefully instead of crashing the UI
        return {"count": 0, "sessions": []}


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    """Delete a session from the active sessions tracking list."""
    if mongo_collection is not None:
        mongo_collection.delete_one({"_id": session_id})
    
    global _active_sessions
    if session_id in _active_sessions:
        del _active_sessions[session_id]
        if mongo_collection is None:
            save_sessions(_active_sessions)
            
    return {"status": "deleted", "session_id": session_id}
