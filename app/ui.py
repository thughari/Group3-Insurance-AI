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

if "force_state_refresh" not in st.session_state:
    st.session_state.force_state_refresh = False


@st.cache_data(ttl=5, show_spinner=False)
def fetch_state(_session_id: str):
    """Fetch copilot state from backend. Cached for 5s to avoid redundant calls on Streamlit reruns."""
    try:
        resp = httpx.get(f"{API_URL}/state/{_session_id}", timeout=10.0)
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
    # If we just switched sessions, bypass cache and do a live fetch
    if st.session_state.force_state_refresh:
        fetch_state.clear()
        st.session_state.force_state_refresh = False
    state_data = fetch_state(st.session_state.session_id)

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

    # Multimodal input has been moved to the chat bar inline.
    
    st.divider()
    if st.button("➕ Start new Session", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.header("📂 Session Manager")
    try:
        sessions_resp = httpx.get(f"{API_URL}/sessions", timeout=5.0)
        if sessions_resp.status_code == 200:
            sessions_data = sessions_resp.json()
            all_sessions = sessions_data.get("sessions", [])

            from datetime import datetime, timedelta

            active_cutoff = datetime.utcnow() - timedelta(minutes=30)
            active_sessions = []
            history_sessions = []
            for s in all_sessions:
                ts = s.get("last_active")
                try:
                    last_active_dt = datetime.fromisoformat(ts)
                except Exception:
                    last_active_dt = None

                is_current = s["session_id"] == st.session_state.session_id
                is_active = is_current or (last_active_dt is not None and last_active_dt >= active_cutoff)
                if is_active:
                    active_sessions.append(s)
                else:
                    history_sessions.append(s)

            st.caption(
                f"{len(active_sessions)} active · {len(history_sessions)} in history"
            )

            def render_session_card(s, section_key: str):
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
                        if st.button("🔀 Switch to this session", key=f"switch_{section_key}_{s['session_id']}", use_container_width=True):
                            # Load conversation history from backend state
                            target_state = {}
                            try:
                                state_resp = httpx.get(f"{API_URL}/state/{s['session_id']}", timeout=5.0)
                                if state_resp.status_code == 200:
                                    target_state = state_resp.json().get("state", {})
                            except Exception:
                                pass
                            st.session_state.session_id = s["session_id"]
                            # Flag that on next render, cache must be cleared for fresh state
                            st.session_state.force_state_refresh = True
                            # Restore conversation history from the backend state
                            history = target_state.get("conversation_history", [])
                            st.session_state.messages = [
                                {"role": msg["role"], "content": msg["content"]}
                                for msg in history
                            ]
                            st.rerun()
                    else:
                        st.success("✅ Current session")

                    if st.button("🗑️ Delete Session", key=f"del_{section_key}_{s['session_id']}", use_container_width=True):
                        try:
                            httpx.delete(f"{API_URL}/sessions/{s['session_id']}", timeout=5.0)
                            if is_current:
                                st.session_state.session_id = str(uuid.uuid4())
                                st.session_state.messages = []
                            st.rerun()
                        except Exception:
                            st.error("Failed to delete session.")

            with st.expander("🟢 Active Sessions", expanded=True):
                if active_sessions:
                    for s in active_sessions:
                        render_session_card(s, "active")
                else:
                    st.caption("No active sessions.")

            with st.expander("🕘 Session History", expanded=False):
                if history_sessions:
                    for s in history_sessions:
                        render_session_card(s, "history")
                else:
                    st.caption("No historical sessions yet.")
        else:
            st.caption("Could not fetch sessions.")
    except Exception:
        st.caption("Backend not reachable.")


# ── Follow-up question mapping (keyword → contextual suggestions) ───────
FOLLOWUP_MAP = {
    "term|plan|option|non-smoker|smoker|age|cover|lakh|crore": [
        "💵 What would the monthly premium be for this plan?",
        "🔄 Can I convert this term plan to a whole life policy later?",
        "📈 What riders (add-ons) should I consider with this plan?",
        "👨‍👩‍👧 How do I nominate beneficiaries for this policy?",
        "🏥 Is a medical exam required for this coverage amount?",
    ],
    "diabetes|blood pressure|pre-existing|health|medical|disease|heart": [
        "💰 How much extra premium would I pay for pre-existing conditions?",
        "📄 What medical reports will I need to submit?",
        "⏳ Is there a waiting period before my conditions are covered?",
        "🔬 Can I get coverage if I manage my condition with medication?",
        "⚖️ What risk tier would I likely fall into?",
    ],
    "beneficiar|spouse|children|nominee|family": [
        "📝 Can I change my beneficiaries after the policy is issued?",
        "⚖️ How is the payout split among multiple beneficiaries?",
        "👶 Can a minor be named as a beneficiary?",
        "🏦 What happens to the policy if the nominee passes away?",
        "📋 What documents are needed to update beneficiaries?",
    ],
    "document|apply|application|paperwork|submit": [
        "🕐 How long does the application process usually take?",
        "🏥 Do I need to do a medical examination?",
        "💳 What are the payment options for premiums?",
        "📱 Can I complete the entire process online?",
        "✅ What happens after I submit my application?",
    ],
    "difference|compare|term|whole life|endowment|ulip": [
        "💰 Which type of insurance gives the best returns?",
        "👨‍👩‍👧 Which plan is best for a family with young children?",
        "📊 Can you compare premiums for term vs whole life for my age?",
        "🎯 What's the best option if I want both protection and savings?",
        "🔄 Can I switch between plan types after purchase?",
    ],
    "risk|underwriting|review|approved|rejected|high-risk|declined": [
        "📋 What factors affect my risk assessment?",
        "🔄 Can I re-apply if my application is declined?",
        "💵 How does my risk tier affect my premium?",
        "🏥 Would improving my health help me get a better rating?",
        "⏳ How long does the underwriting process take?",
    ],
    "premium|cost|price|payment|afford": [
        "📅 What premium payment frequencies are available (monthly, yearly)?",
        "💳 Can I pay premiums via auto-debit or UPI?",
        "📉 How can I reduce my premium amount?",
        "⚠️ What happens if I miss a premium payment?",
        "🎁 Are there any tax benefits on the premium I pay?",
    ],
}

GENERIC_FOLLOWUPS = [
    "📋 What documents do I need to apply?",
    "💰 How are premiums calculated for my profile?",
    "👨‍👩‍👧‍👦 How do I add beneficiaries to my policy?",
    "⚖️ What types of life insurance are available?",
    "🏥 Is a medical examination required?",
]

import re

def get_followup_questions(last_user_msg: str, n: int = 3) -> list[str]:
    """Pick the most relevant follow-up questions based on the last user message."""
    if not last_user_msg:
        return GENERIC_FOLLOWUPS[:n]

    msg_lower = last_user_msg.lower()
    best_match = None
    best_score = 0

    for pattern, questions in FOLLOWUP_MAP.items():
        keywords = pattern.split("|")
        score = sum(1 for kw in keywords if kw in msg_lower)
        if score > best_score:
            best_score = score
            best_match = questions

    if best_match:
        return best_match[:n]
    return GENERIC_FOLLOWUPS[:n]


# ── Chat UI ─────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Quick Demo Questions (shown only when chat is empty) ────────────────
DEMO_QUESTIONS = [
    "🧑‍💼 I'm a 30-year-old non-smoker looking for a ₹50 lakh term plan for 20 years. What are my options?",
    "🏥 I have diabetes and high blood pressure. Can I still get life insurance?",
    "👨‍👩‍👧‍👦 How do I add my spouse and children as beneficiaries to my policy?",
    "📋 What documents do I need to apply for a life insurance policy?",
    "⚖️ What's the difference between term insurance and whole life insurance?",
    "💰 I want to apply for a ₹1 crore policy. I'm 45, smoker, with a history of heart disease.",
]

# Custom CSS for the suggestion chips (used for both welcome & follow-ups)
st.markdown("""
<style>
div[data-testid="stVerticalBlock"] div.quick-q-header {
    text-align: center;
    margin-bottom: 0.5rem;
}
div.followup-header {
    margin-top: 0.25rem;
    margin-bottom: 0.25rem;
}
div.followup-header p {
    font-size: 0.85rem;
    color: #888;
}
/* Style the demo / follow-up question buttons */
div[data-testid="stVerticalBlock"] button[kind="secondary"] {
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    border-radius: 12px !important;
    transition: all 0.2s ease !important;
    font-size: 0.85rem !important;
}
div[data-testid="stVerticalBlock"] button[kind="secondary"]:hover {
    border-color: rgba(99, 102, 241, 0.7) !important;
    background-color: rgba(99, 102, 241, 0.08) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.15) !important;
}
</style>
""", unsafe_allow_html=True)

if not st.session_state.messages:
    # ── Welcome screen ──
    st.markdown(
        "<div class='quick-q-header'>"
        "<h3>👋 Welcome! Try one of these questions to get started:</h3>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Render questions in a 2-column grid
    cols = st.columns(2)
    for i, question in enumerate(DEMO_QUESTIONS):
        with cols[i % 2]:
            if st.button(question, key=f"demo_q_{i}", use_container_width=True):
                st.session_state.pending_prompt = question
                st.rerun()

else:
    # ── Follow-up suggestions after the last assistant response ──
    # Find the last user message to determine context
    last_user_msg = ""
    for msg in reversed(st.session_state.messages):
        if msg["role"] == "user":
            last_user_msg = msg["content"]
            break

    # Only show follow-ups if the last message is from the assistant
    if st.session_state.messages[-1]["role"] == "assistant":
        followups = get_followup_questions(last_user_msg)

        st.markdown(
            "<div class='followup-header'><p>💡 <b>Suggested follow-ups:</b></p></div>",
            unsafe_allow_html=True,
        )
        cols = st.columns(len(followups))
        for i, fq in enumerate(followups):
            with cols[i]:
                if st.button(fq, key=f"followup_{i}", use_container_width=True):
                    st.session_state.pending_prompt = fq
                    st.rerun()

prompt_input = st.chat_input(
    "Ask a question or attach a file...", 
    accept_file=True, 
    file_type=["png", "jpg", "jpeg", "wav", "mp3", "m4a", "ogg", "webm", "flac"]
)

prompt = None
if prompt_input:
    # If accept_file=True, chat_input returns an object with .text and .files attributes
    prompt_text = getattr(prompt_input, "text", "")
    attached_files = getattr(prompt_input, "files", [])
    
    if attached_files:
        with st.spinner("Processing attached media..."):
            gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            openai_key = os.getenv("OPENAI_API_KEY")
            if not gemini_key and not openai_key:
                st.error("Gemini or OpenAI API key is required for Multimodal input.")
            else:
                try:
                    transcribed_text = ""
                    if gemini_key:
                        client = genai.Client(api_key=gemini_key)
                        contents = ["Extract and transcribe the text from the provided image or audio. Output only the transcribed text/query."]
                        for f in attached_files:
                            mime = f.type
                            if mime.startswith("audio/"):
                                contents.append(types.Part.from_bytes(data=f.getvalue(), mime_type=mime))
                            elif mime.startswith("image/"):
                                contents.append(types.Part.from_bytes(data=f.getvalue(), mime_type=mime))
                        
                        response = client.models.generate_content(model='gemini-2.5-flash', contents=contents)
                        transcribed_text = response.text.strip()
                    elif openai_key:
                        from openai import OpenAI
                        import base64
                        client = OpenAI(api_key=openai_key)
                        for f in attached_files:
                            mime = f.type
                            if mime.startswith("audio/"):
                                audio_response = client.audio.transcriptions.create(
                                    model="whisper-1", 
                                    file=(f.name, f.getvalue())
                                )
                                transcribed_text += audio_response.text + "\n"
                            elif mime.startswith("image/"):
                                base64_image = base64.b64encode(f.getvalue()).decode('utf-8')
                                image_url = f"data:{mime};base64,{base64_image}"
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
                                transcribed_text += response.choices[0].message.content.strip() + "\n"
                        transcribed_text = transcribed_text.strip()
                    
                    if transcribed_text:
                        prompt = f"{prompt_text}\n\n[Transcribed Media Context:\n{transcribed_text}]".strip()
                    else:
                        prompt = prompt_text
                except Exception as e:
                    st.error(f"Error processing media: {e}")
                    prompt = prompt_text
    else:
        prompt = prompt_text

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
