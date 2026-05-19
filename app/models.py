import operator
from typing import Any, Dict, List, Literal, Optional, Annotated

from pydantic import BaseModel, Field


Intent = Literal["underwriting", "policy_qa", "beneficiary", "issuance", "lapse_revival", "policy_comparison", "lapse_prediction"]
RiskTier = Literal["standard", "substandard", "high", "declined", "unknown"]


# --- API request/response schemas (Pydantic) ---

class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    session_id: str
    node_path: List[str]
    response: str
    state: Dict[str, Any]


# --- LangGraph state (TypedDict with reducers) ---
# Using Annotated with operator.add for list fields ensures that
# each node APPENDS to the list rather than overwriting it.

from typing import TypedDict
import operator

def add_and_truncate_history(left: List[Dict[str, str]], right: List[Dict[str, str]]) -> List[Dict[str, str]]:
    left = left or []
    right = right or []
    # Combine and keep only the last 10 messages to save context window
    return (left + right)[-10:]

class CopilotState(TypedDict, total=False):
    session_id: str
    user_query: str
    intent: Optional[Intent]
    response: str
    applicant_data: Dict[str, Any]
    risk_tier: RiskTier
    policy_type_preference: Optional[str]
    conversation_history: Annotated[List[Dict[str, str]], add_and_truncate_history]
    node_outputs: Dict[str, Any]
    requires_human_review: bool
    approved_by_human: Optional[bool]
    node_path: Annotated[List[str], operator.add]
