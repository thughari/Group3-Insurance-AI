import json
import streamlit as st
import httpx
import uuid
import os
from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path=env_path, override=True)

from google import genai
from google.genai import types

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Life Insurance AI Copilot", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []


def fetch_state():
    try:
        resp = httpx.get(f"{API_URL}/state/{st.session_state.session_id}", timeout=10.0)
        if resp.status_code == 200:
            return resp.json().get("state", {})
    except Exception:
        return {}
    return {}


def stream_chat(message: str):
    """
    Calls the /chat/stream SSE endpoint and yields tokens.
    Falls back to /chat if streaming fails.
    """
    try:
        with httpx.stream(
            "POST",
            f"{API_URL}/chat/stream",
            json={"session_id": st.session_state.session_id, "message": message},
            timeout=60.0,
        ) as response:
            meta = None
            full_text = ""
            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]  # strip "data: "
                if payload == "[DONE]":
                    break
                try:
                    data = json.loads(payload)
                    if data.get("type") == "meta":
                        meta = data
                    elif data.get("type") == "token":
                        chunk = data.get("content", "")
                        full_text += chunk
                        yield chunk
                    elif data.get("type") == "blocked":
                        yield data.get("content", "Blocked by guardrails.")
                        return
                    elif data.get("type") == "paused":
                        yield data.get("content", "Application paused for review.")
                        return
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        # Fallback to non-streaming
        try:
            res = httpx.post(
                f"{API_URL}/chat",
                json={"session_id": st.session_state.session_id, "message": message},
                timeout=30.0,
            )
            if res.status_code == 200:
                yield res.json().get("response", "No response")
            else:
                yield f"Error: {res.status_code}"
        except Exception as e2:
            yield f"Connection failed: {e2}"


# ── Page title ──────────────────────────────────────────────────────────
st.title("🛡️ Life Insurance AI Copilot")
st.caption("Powered by LangGraph · Ask about policies, underwriting, beneficiaries, or issuance")

# ── Sidebar for state display and HitL ──────────────────────────────────
with st.sidebar:
    st.header("📊 Copilot State")
    state_data = fetch_state()

    if state_data:
        app_data = state_data.get("applicant_data", {})
        if app_data:
            st.subheader("Applicant Data")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Age", app_data.get("age", "N/A"))
                st.metric("Term", f"{app_data.get('term_years', 'N/A')} yrs")
            with col2:
                cover = app_data.get("cover_amount", "N/A")
                if isinstance(cover, (int, float)):
                    st.metric("Cover", f"₹{cover:,.0f}")
                else:
                    st.metric("Cover", cover)
            disclosures = app_data.get("health_disclosures", [])
            if disclosures:
                st.write(f"**Disclosures:** {', '.join(disclosures)}")

        risk_tier = state_data.get("risk_tier", "unknown")
        if risk_tier != "unknown":
            st.subheader("Underwriting")
            color = {"standard": "🟢", "substandard": "🟡", "high": "🔴", "declined": "⛔"}.get(risk_tier, "⚪")
            st.write(f"**Risk Tier:** {color} {risk_tier.upper()}")

        node_path = state_data.get("node_path", [])
        if node_path:
            st.subheader("Execution Trace")
            st.write(" ➔ ".join(node_path))

        # HitL approval logic
        if state_data.get("is_paused"):
            st.error("⚠️ Human Review Required")
            st.write("A human underwriter must approve or reject this application before proceeding.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Approve", use_container_width=True):
                    res = httpx.post(
                        f"{API_URL}/approve",
                        json={"session_id": st.session_state.session_id, "approved": True},
                        timeout=30.0,
                    )
                    if res.status_code == 200:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "✅ Underwriter decision: **Approved**. Your policy application will proceed."
                        })
                        st.rerun()
            with col2:
                if st.button("❌ Reject", use_container_width=True):
                    res = httpx.post(
                        f"{API_URL}/approve",
                        json={"session_id": st.session_state.session_id, "approved": False},
                        timeout=30.0,
                    )
                    if res.status_code == 200:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "❌ Underwriter decision: **Rejected**. We cannot proceed with the policy at this time."
                        })
                        st.rerun()

    st.divider()
    st.header("🎙️ Multimodal Input")
    st.caption("Upload a voice recording or image (e.g. medical report) to chat.")
    audio_val = st.file_uploader("🎤 Upload Voice Recording", type=["wav", "mp3", "m4a", "ogg", "webm", "flac"])
    image_val = st.file_uploader("🖼️ Upload Image", type=["png", "jpg", "jpeg"])
    if st.button("Submit Media", use_container_width=True):
        if audio_val or image_val:
            with st.spinner("Processing media..."):
                gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
                openai_key = os.getenv("OPENAI_API_KEY")
                if not gemini_key and not openai_key:
                    st.error("Gemini or OpenAI API key is required for Multimodal input.")
                else:
                    try:
                        transcribed_text = ""
                        # Extract raw audio bytes
                        raw_audio = audio_val.getvalue() if audio_val else None
                        audio_fname = getattr(audio_val, "name", "audio.wav") if audio_val else None
                        
                        if gemini_key:
                            client = genai.Client(api_key=gemini_key)
                            contents = ["Extract and transcribe the text from the provided image or audio. Output only the transcribed text/query."]
                            if raw_audio:
                                contents.append(types.Part.from_bytes(data=raw_audio, mime_type="audio/wav"))
                            if image_val:
                                contents.append(types.Part.from_bytes(data=image_val.getvalue(), mime_type=image_val.type))
                            response = client.models.generate_content(model='gemini-2.5-flash', contents=contents)
                            transcribed_text = response.text.strip()
                        elif openai_key:
                            from openai import OpenAI
                            import base64
                            client = OpenAI(api_key=openai_key)
                            if raw_audio:
                                audio_response = client.audio.transcriptions.create(
                                    model="whisper-1", 
                                    file=(audio_fname, raw_audio)
                                )
                                transcribed_text += audio_response.text + "\n"
                            if image_val:
                                base64_image = base64.b64encode(image_val.getvalue()).decode('utf-8')
                                image_url = f"data:{image_val.type};base64,{base64_image}"
                                response = client.chat.completions.create(
                                    model="gpt-4o-mini",
                                    messages=[
                                        {
                                            "role": "user",
                                            "content": [
                                                {"type": "text", "text": "Extract and transcribe the text from this image. Output only the transcribed text/query."},
                                                {"type": "image_url", "image_url": {"url": image_url}}
                                            ]
                                        }
                                    ]
                                )
                                transcribed_text += response.choices[0].message.content.strip()
                            transcribed_text = transcribed_text.strip()
                        
                        if transcribed_text:
                            st.session_state.pending_prompt = transcribed_text
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error processing media: {e}")

    st.divider()
    st.header("📋 Active Sessions")
    try:
        sessions_resp = httpx.get(f"{API_URL}/sessions", timeout=5.0)
        if sessions_resp.status_code == 200:
            sessions_data = sessions_resp.json()
            st.caption(f"{sessions_data['count']} session(s) active")
            for s in sessions_data.get("sessions", []):
                is_current = s["session_id"] == st.session_state.session_id
                label = f"{'🟢' if is_current else '⚪'} `{s['session_id'][:8]}...`"
                with st.expander(label, expanded=is_current):
                    st.write(f"**Last Query:** {s.get('last_query', 'N/A')}")
                    st.write(f"**Intent:** `{s.get('intent', 'N/A')}`")
                    st.write(f"**Trace:** {' ➔ '.join(s.get('node_path', []))}")
                    if s.get("is_paused"):
                        st.warning("⚠️ Paused for HitL review")
                    st.caption(f"Last active: {s.get('last_active', 'N/A')}")
                    if not is_current:
                        if st.button("🔀 Switch to this session", key=f"switch_{s['session_id']}", use_container_width=True):
                            # Load conversation history from backend state
                            target_state = {}
                            try:
                                state_resp = httpx.get(f"{API_URL}/state/{s['session_id']}", timeout=5.0)
                                if state_resp.status_code == 200:
                                    target_state = state_resp.json().get("state", {})
                            except Exception:
                                pass
                            st.session_state.session_id = s["session_id"]
                            # Restore conversation history from the backend state
                            history = target_state.get("conversation_history", [])
                            st.session_state.messages = [
                                {"role": msg["role"], "content": msg["content"]}
                                for msg in history
                            ]
                            st.rerun()
                    else:
                        st.success("✅ Current session")
                    
                    if st.button("🗑️ Delete Session", key=f"del_{s['session_id']}", use_container_width=True):
                        try:
                            httpx.delete(f"{API_URL}/sessions/{s['session_id']}", timeout=5.0)
                            if is_current:
                                st.session_state.session_id = str(uuid.uuid4())
                                st.session_state.messages = []
                            st.rerun()
                        except Exception:
                            st.error("Failed to delete session.")
        else:
            st.caption("Could not fetch sessions.")
    except Exception:
        st.caption("Backend not reachable.")

    st.divider()
    if st.button("🗑️ Clear Chat & Reset Session", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

# ── Chat UI ─────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask a question about life insurance...")
if "pending_prompt" in st.session_state and st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    del st.session_state.pending_prompt

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing your request..."):
            # Stream the response
            full_response = st.write_stream(stream_chat(prompt))
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    st.rerun()
