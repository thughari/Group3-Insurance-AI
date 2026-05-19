import os
from typing import Dict, Literal, Optional, AsyncIterator

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessageChunk
from pydantic import BaseModel, Field

from app.models import CopilotState
from app.tools.csv_lookup import classify_risk, indicative_premium_lookup
from app.tools.rag import retrieve_policy_context


# ── LLM provider selector ──────────────────────────────────────────────

from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache

set_llm_cache(InMemoryCache())
print("✅ In-Memory LLM Cache Enabled")

def get_llm():
    groq_key = os.getenv("GROQ_API_KEY")
    gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if groq_key:
        from langchain_groq import ChatGroq
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=groq_key,
        )
    elif gemini_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=gemini_key,
        )
    elif openai_key:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    else:
        raise ValueError(
            "No LLM API key found. Set GROQ_API_KEY, GOOGLE_API_KEY, or OPENAI_API_KEY in your .env file."
        )


# ── Structured output schemas ───────────────────────────────────────────

class IntentClassification(BaseModel):
    intent: Literal["underwriting", "policy_qa", "beneficiary", "issuance", "lapse_revival", "policy_comparison", "lapse_prediction"] = Field(
        description="Classify the user query into one of the allowed intents."
    )


class ApplicantDataExtract(BaseModel):
    age: Optional[int] = Field(default=None, description="Age of the applicant if mentioned.")
    cover_amount: Optional[int] = Field(default=None, description="Desired cover amount or sum assured if mentioned.")
    term_years: Optional[int] = Field(default=None, description="Policy term in years if mentioned.")
    health_disclosures: list[str] = Field(default_factory=list, description="Any health conditions, smoking habits, or lifestyle factors mentioned.")


# ── Helpers ─────────────────────────────────────────────────────────────

def format_history(history: list) -> str:
    if not history:
        return "No previous conversation."
    return "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-4:]])


# ── Graph nodes ─────────────────────────────────────────────────────────
# All nodes accept CopilotState (TypedDict) and return a dict of updates.
# List fields with Annotated[..., operator.add] reducers (node_path,
# conversation_history) are APPENDED to, so we return only the NEW items.

async def intent_router(state: CopilotState) -> Dict:
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an intent classifier for a life insurance copilot. "
         "Classify the user's intent based on their latest query and conversation history.\n"
         "- underwriting: asking about risk, premium estimates, health disclosures, cover amount.\n"
         "- policy_qa: asking about product types (term vs whole), coverages, riders.\n"
         "- beneficiary: asking about nominations, nominees, shares.\n"
         "- issuance: asking about pending documents, application status, issuance timelines.\n"
         "- lapse_revival: asking about missed premiums, grace periods, reinstatement, revival.\n"
         "- policy_comparison: explicitly asking to compare policy types (e.g., term vs whole life).\n"
         "- lapse_prediction: asking to predict lapse risk based on payment history.\n"
         "Default to policy_qa if unsure.\n\nConversation History:\n{history}"),
        ("user", "{query}")
    ])

    history = state.get("conversation_history", [])
    query = state.get("user_query", "")

    try:
        structured_llm = llm.with_structured_output(IntentClassification)
        result = await structured_llm.ainvoke(prompt.format_prompt(
            history=format_history(history), query=query
        ))
        intent = result.intent
    except Exception as e:
        print(f"LLM Routing failed: {e}")
        # Fallback rule-based
        q = query.lower()
        if any(k in q for k in ["underwriting", "premium", "risk", "smoker", "diabetes", "cover", "age"]):
            intent = "underwriting"
        elif any(k in q for k in ["beneficiary", "nominee", "share"]):
            intent = "beneficiary"
        elif any(k in q for k in ["issuance", "pending", "document", "timeline", "status"]):
            intent = "issuance"
        elif any(k in q for k in ["lapse", "revival", "reinstat", "grace period", "missed premium"]):
            intent = "lapse_revival"
        elif "compare" in q or "vs" in q or "difference" in q:
            intent = "policy_comparison"
        elif "predict" in q or "history" in q:
            intent = "lapse_prediction"
        else:
            intent = "policy_qa"

    return {"intent": intent, "node_path": ["intent_router"]}


async def underwriting_agent(state: CopilotState) -> Dict:
    llm = get_llm()
    query = state.get("user_query", "")

    # Extract data from current query
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract applicant data from the user query. Only extract what is explicitly mentioned."),
        ("user", "{query}")
    ])

    try:
        structured_llm = llm.with_structured_output(ApplicantDataExtract)
        extracted = await structured_llm.ainvoke(prompt.format_prompt(query=query))

        # Merge with existing state
        data = dict(state.get("applicant_data", {}))
        if extracted.age:
            data["age"] = extracted.age
        if extracted.cover_amount:
            data["cover_amount"] = extracted.cover_amount
        if extracted.term_years:
            data["term_years"] = extracted.term_years
        if extracted.health_disclosures:
            existing_health = data.get("health_disclosures", [])
            new_health = [d for d in extracted.health_disclosures if d not in existing_health]
            data["health_disclosures"] = existing_health + new_health
    except Exception as e:
        print(f"Extraction failed: {e}")
        data = dict(state.get("applicant_data", {}))

    disclosures = data.get("health_disclosures", [])
    age = int(data.get("age") or 35)
    cover = int(data.get("cover_amount") or 1000000)
    term = int(data.get("term_years") or 20)

    risk_tier = classify_risk(disclosures)
    estimate = indicative_premium_lookup(age, cover, term, risk_tier)

    requires_human = risk_tier in {"high", "substandard", "declined"}

    # Generate natural language response
    response_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an underwriting assistant. Summarize the applicant's current status and premium estimate. "
         "Never provide a final underwriting decision. Emphasize estimates are non-binding and indicative only. "
         "If risk tier is high or substandard, mention it requires human underwriter review."),
        ("user",
         "Data: Age {age}, Cover {cover}, Term {term}, Conditions: {disclosures}.\n"
         "Risk Tier: {risk_tier}. Estimate: {estimate_amt}\n"
         "User query: {query}")
    ])

    resp_msg = await llm.ainvoke(response_prompt.format_prompt(
        age=age, cover=cover, term=term, disclosures=disclosures,
        risk_tier=risk_tier, estimate_amt=estimate["monthly_estimate"],
        query=query
    ), config={"tags": ["final_response"]})

    return {
        "applicant_data": data,
        "risk_tier": risk_tier,
        "requires_human_review": requires_human,
        "response": resp_msg.content,
        "node_outputs": {**state.get("node_outputs", {}), "underwriting": {"estimate": estimate}},
        "node_path": ["underwriting_agent"],
    }


async def policy_qa_agent(state: CopilotState) -> Dict:
    llm = get_llm()
    query = state.get("user_query", "")
    history = state.get("conversation_history", [])

    # Improve RAG search for short/vague queries
    search_query = query
    if history and len(query.split()) < 6:
        search_query = f"{history[-1]['content']} {query}"

    ctx = retrieve_policy_context(search_query)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a Life Insurance Copilot answering policy questions. "
         "Use the provided context to answer. Always include citations "
         "(Document name and Page number) from the context. "
         "If the context does not contain the answer, say you don't know based on the documents.\n\n"
         "Context:\n{context}\n\nConversation History:\n{history}"),
        ("user", "{query}")
    ])

    resp = await llm.ainvoke(prompt.format_prompt(
        context=ctx, history=format_history(history), query=query
    ), config={"tags": ["final_response"]})

    return {
        "response": resp.content,
        "node_path": ["policy_qa_agent"],
    }


async def beneficiary_agent(state: CopilotState) -> Dict:
    llm = get_llm()
    query = state.get("user_query", "")
    ctx = retrieve_policy_context("Beneficiary nomination " + query)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a Beneficiary agent. Guide the user on nomination rules, share allocations, "
         "and minor nominee rules based on the context. Always include citations "
         "(Document name and Page number).\n\nContext:\n{context}"),
        ("user", "{query}")
    ])

    resp = await llm.ainvoke(prompt.format_prompt(context=ctx, query=query), config={"tags": ["final_response"]})

    return {
        "response": resp.content,
        "node_path": ["beneficiary_agent"],
    }


async def issuance_agent(state: CopilotState) -> Dict:
    llm = get_llm()
    query = state.get("user_query", "")
    ctx = retrieve_policy_context("Policy issuance status documents " + query)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an Issuance agent. Answer questions about pending documents and issuance timelines "
         "based on the context. Always include citations (Document name and Page number).\n\nContext:\n{context}"),
        ("user", "{query}")
    ])

    resp = await llm.ainvoke(prompt.format_prompt(context=ctx, query=query), config={"tags": ["final_response"]})

    return {
        "response": resp.content,
        "node_path": ["issuance_agent"],
    }


async def lapse_revival_agent(state: CopilotState) -> Dict:
    """Dedicated agent for lapse, revival, and reinstatement queries."""
    llm = get_llm()
    query = state.get("user_query", "")
    ctx = retrieve_policy_context("Lapse revival reinstatement grace period missed premium " + query)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a Lapse & Revival specialist agent for life insurance. "
         "Answer questions about missed premiums, grace periods, policy lapse conditions, "
         "revival requirements, reinstatement documentation, and surrender value implications. "
         "Always include citations (Document name and Page number).\n\nContext:\n{context}"),
        ("user", "{query}")
    ])

    resp = await llm.ainvoke(prompt.format_prompt(context=ctx, query=query), config={"tags": ["final_response"]})

    return {
        "response": resp.content,
        "node_path": ["lapse_revival_agent"],
    }


async def policy_comparison_agent(state: CopilotState) -> Dict:
    llm = get_llm()
    query = state.get("user_query", "")
    ctx = retrieve_policy_context("Compare life insurance policy types pros and cons " + query)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a Policy Comparison specialist. The user wants to compare different policy types. "
         "Using the provided context, generate a structured Markdown table comparing the requested policies "
         "highlighting their pros and cons. Always include citations (Document name and Page number) below the table.\n\nContext:\n{context}"),
        ("user", "{query}")
    ])

    resp = await llm.ainvoke(prompt.format_prompt(context=ctx, query=query), config={"tags": ["final_response"]})
    return {
        "response": resp.content,
        "node_path": ["policy_comparison_agent"],
    }


async def lapse_prediction_agent(state: CopilotState) -> Dict:
    llm = get_llm()
    query = state.get("user_query", "")
    ctx = retrieve_policy_context("Lapse revival reinstatement " + query)
    
    # Mocking payment history check
    mock_payment_history = "User has missed 2 consecutive premium payments in the last 6 months."
    
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a Lapse Prediction specialist. Evaluate the lapse risk based on the user's payment history. "
         "Proactively suggest revival options if the risk is high. Use the provided policy context for accurate rules. "
         "Always include citations.\n\nContext:\n{context}\n\nPayment History:\n{payment_history}"),
        ("user", "{query}")
    ])

    resp = await llm.ainvoke(prompt.format_prompt(context=ctx, payment_history=mock_payment_history, query=query), config={"tags": ["final_response"]})
    return {
        "response": resp.content,
        "node_path": ["lapse_prediction_agent"],
    }


def human_review(state: CopilotState) -> Dict:
    """Adds an explicit pause/review message when risk is elevated."""
    current_response = state.get("response", "")
    return {
        "response": current_response + "\n\n⚠️ **[SYSTEM: High-risk/substandard case. PAUSING for Human Underwriter review. Please approve or reject in the sidebar.]**",
        "node_path": ["human_review"],
    }


# ── Streaming helpers ───────────────────────────────────────────────────

async def stream_agent_response(state: CopilotState, agent_name: str) -> AsyncIterator[str]:
    """
    Streams the LLM response token-by-token for the given agent.
    Returns an async iterator of string chunks.
    """
    llm = get_llm()
    query = state.get("user_query", "")
    history = state.get("conversation_history", [])

    if agent_name == "policy_qa_agent":
        search_query = query
        if history and len(query.split()) < 6:
            search_query = f"{history[-1]['content']} {query}"
        ctx = retrieve_policy_context(search_query)
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a Life Insurance Copilot answering policy questions. "
             "Use the provided context to answer. Always include citations "
             "(Document name and Page number) from the context.\n\n"
             "Context:\n{context}\n\nConversation History:\n{history}"),
            ("user", "{query}")
        ])
        messages = prompt.format_prompt(context=ctx, history=format_history(history), query=query).to_messages()

    elif agent_name == "beneficiary_agent":
        ctx = retrieve_policy_context("Beneficiary nomination " + query)
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a Beneficiary agent. Guide on nomination rules, share allocations, "
             "and minor nominee rules. Include citations.\n\nContext:\n{context}"),
            ("user", "{query}")
        ])
        messages = prompt.format_prompt(context=ctx, query=query).to_messages()

    elif agent_name == "issuance_agent":
        ctx = retrieve_policy_context("Policy issuance status documents " + query)
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an Issuance agent. Answer about pending documents and timelines. "
             "Include citations.\n\nContext:\n{context}"),
            ("user", "{query}")
        ])
        messages = prompt.format_prompt(context=ctx, query=query).to_messages()

    elif agent_name == "lapse_revival_agent":
        ctx = retrieve_policy_context("Lapse revival reinstatement grace period " + query)
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a Lapse & Revival specialist. Answer about missed premiums, "
             "grace periods, revival requirements. Include citations.\n\nContext:\n{context}"),
            ("user", "{query}")
        ])
        messages = prompt.format_prompt(context=ctx, query=query).to_messages()

    elif agent_name == "policy_comparison_agent":
        ctx = retrieve_policy_context("Compare life insurance policy types pros and cons " + query)
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a Policy Comparison specialist. Generate a structured Markdown table comparing the requested policies "
             "highlighting their pros and cons. Include citations.\n\nContext:\n{context}"),
            ("user", "{query}")
        ])
        messages = prompt.format_prompt(context=ctx, query=query).to_messages()

    elif agent_name == "lapse_prediction_agent":
        ctx = retrieve_policy_context("Lapse revival reinstatement " + query)
        mock_payment_history = "User has missed 2 consecutive premium payments in the last 6 months."
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a Lapse Prediction specialist. Evaluate lapse risk based on payment history and suggest revival options. "
             "Include citations.\n\nContext:\n{context}\n\nPayment History:\n{payment_history}"),
            ("user", "{query}")
        ])
        messages = prompt.format_prompt(context=ctx, payment_history=mock_payment_history, query=query).to_messages()

    elif agent_name == "underwriting_agent":
        # Underwriting has structured extraction first, so we stream only the final response
        data = dict(state.get("applicant_data", {}))
        disclosures = data.get("health_disclosures", [])
        age = int(data.get("age") or 35)
        cover = int(data.get("cover_amount") or 1000000)
        term = int(data.get("term_years") or 20)
        risk_tier = state.get("risk_tier", "unknown")
        node_outputs = state.get("node_outputs", {})
        estimate = node_outputs.get("underwriting", {}).get("estimate", {})
        estimate_amt = estimate.get("monthly_estimate", "N/A")

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an underwriting assistant. Summarize the applicant's status and premium estimate. "
             "Never provide a final decision. Emphasize estimates are non-binding."),
            ("user",
             "Data: Age {age}, Cover {cover}, Term {term}, Conditions: {disclosures}.\n"
             "Risk Tier: {risk_tier}. Estimate: {estimate_amt}\nUser query: {query}")
        ])
        messages = prompt.format_prompt(
            age=age, cover=cover, term=term, disclosures=disclosures,
            risk_tier=risk_tier, estimate_amt=estimate_amt, query=query
        ).to_messages()
    else:
        yield "Agent not available for streaming."
        return

    async for chunk in llm.astream(messages):
        if isinstance(chunk, AIMessageChunk) and chunk.content:
            yield chunk.content
        elif hasattr(chunk, "content") and chunk.content:
            yield chunk.content


# ── Routing functions ───────────────────────────────────────────────────

def route_from_intent(state: CopilotState) -> str:
    return state.get("intent") or "policy_qa"


def route_from_underwriting(state: CopilotState) -> str:
    return "human_review" if state.get("requires_human_review") else "end"


# ── Graph builder ───────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(CopilotState)

    graph.add_node("intent_router", intent_router)
    graph.add_node("underwriting_agent", underwriting_agent)
    graph.add_node("policy_qa_agent", policy_qa_agent)
    graph.add_node("beneficiary_agent", beneficiary_agent)
    graph.add_node("issuance_agent", issuance_agent)
    graph.add_node("lapse_revival_agent", lapse_revival_agent)
    graph.add_node("policy_comparison_agent", policy_comparison_agent)
    graph.add_node("lapse_prediction_agent", lapse_prediction_agent)
    graph.add_node("human_review", human_review)

    graph.add_edge(START, "intent_router")
    graph.add_conditional_edges(
        "intent_router",
        route_from_intent,
        {
            "underwriting": "underwriting_agent",
            "policy_qa": "policy_qa_agent",
            "beneficiary": "beneficiary_agent",
            "issuance": "issuance_agent",
            "lapse_revival": "lapse_revival_agent",
            "policy_comparison": "policy_comparison_agent",
            "lapse_prediction": "lapse_prediction_agent",
        },
    )
    graph.add_conditional_edges(
        "underwriting_agent",
        route_from_underwriting,
        {"human_review": "human_review", "end": END},
    )
    graph.add_edge("policy_qa_agent", END)
    graph.add_edge("beneficiary_agent", END)
    graph.add_edge("issuance_agent", END)
    graph.add_edge("lapse_revival_agent", END)
    graph.add_edge("policy_comparison_agent", END)
    graph.add_edge("lapse_prediction_agent", END)
    graph.add_edge("human_review", END)

    import os
    mongodb_uri = os.getenv("MONGODB_URI")
    
    if mongodb_uri:
        import pymongo
        import certifi
        from langgraph.checkpoint.mongodb import MongoDBSaver
        client = pymongo.MongoClient(mongodb_uri, tlsCAFile=certifi.where())
        memory = MongoDBSaver(client)
    else:
        from langgraph.checkpoint.memory import MemorySaver
        memory = MemorySaver()
        print("WARNING: MONGODB_URI not set. Falling back to in-memory checkpointer. HitL and persistence will be lost on restart.")
    
    return graph.compile(checkpointer=memory, interrupt_before=["human_review"])
