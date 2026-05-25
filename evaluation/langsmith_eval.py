import os
import time
import httpx
from dotenv import load_dotenv
load_dotenv()

from langsmith import Client, evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)
from deepeval.test_case import LLMTestCase

# Initialize LangSmith client
client = Client()

def predict_copilot(inputs: dict) -> dict:
    """Target function to run against each dataset example."""
    # Handle different dataset schemas
    question = inputs.get("question") or inputs.get("input") or list(inputs.values())[0]
    session_id = f"ls-eval-{int(time.time()*1000)}"
    
    api_url = os.getenv("API_URL", "http://localhost:8000")
    
    try:
        resp = httpx.post(
            f"{api_url}/chat",
            json={"session_id": session_id, "message": question},
            timeout=60.0,
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            return {"response": f"ERROR: {resp.status_code}", "state": {}}
    except Exception as e:
        return {"response": f"CONNECTION_ERROR: {e}", "state": {}}

def faithfulness_evaluator(run, example) -> dict:
    """Evaluate Faithfulness using DeepEval."""
    inputs = example.inputs
    question = inputs.get("question") or inputs.get("input") or list(inputs.values())[0]
    
    outputs = run.outputs or {}
    response = outputs.get("response", "")
    state = outputs.get("state", {})
    retrieval_context = state.get("retrieval_context", "")
    
    # If no context was retrieved (e.g., underwriting agent), we skip faithfulness 
    # since Faithfulness checks if the answer matches the documents.
    if not retrieval_context:
        return {"key": "faithfulness", "score": None, "comment": "Skipped: No RAG context retrieved for this query"}
        
    test_case = LLMTestCase(
        input=question,
        actual_output=response,
        retrieval_context=[retrieval_context],
    )
    metric = FaithfulnessMetric(threshold=0.85)
    try:
        metric.measure(test_case)
        return {"key": "faithfulness", "score": metric.score, "comment": metric.reason}
    except Exception as e:
        return {"key": "faithfulness", "score": 0, "comment": str(e)}

def relevancy_evaluator(run, example) -> dict:
    """Evaluate Answer Relevancy using DeepEval."""
    inputs = example.inputs
    question = inputs.get("question") or inputs.get("input") or list(inputs.values())[0]
    
    outputs = run.outputs or {}
    response = outputs.get("response", "")
    
    test_case = LLMTestCase(
        input=question,
        actual_output=response,
        retrieval_context=[], # Not strictly required for relevancy
    )
    metric = AnswerRelevancyMetric(threshold=0.80)
    try:
        metric.measure(test_case)
        return {"key": "answer_relevancy", "score": metric.score, "comment": metric.reason}
    except Exception as e:
        return {"key": "answer_relevancy", "score": 0, "comment": str(e)}

def relevance_evaluator(run, example) -> dict:
    """Evaluate Answer Relevance using DeepEval."""
    inputs = example.inputs
    question = inputs.get("question") or inputs.get("input") or list(inputs.values())[0]
    
    outputs = run.outputs or {}
    response = outputs.get("response", "")
    
    test_case = LLMTestCase(
        input=question,
        actual_output=response,
        retrieval_context=[],
    )
    metric = AnswerRelevancyMetric(threshold=0.80)
    try:
        metric.measure(test_case)
        return {"key": "answer_relevance", "score": metric.score, "comment": metric.reason}
    except Exception as e:
        return {"key": "answer_relevance", "score": 0, "comment": str(e)}

def contextual_precision_evaluator(run, example) -> dict:
    """Evaluate Contextual Precision using DeepEval."""
    inputs = example.inputs
    question = inputs.get("question") or inputs.get("input") or list(inputs.values())[0]
    
    outputs = run.outputs or {}
    response = outputs.get("response", "")
    state = outputs.get("state", {})
    retrieval_context = state.get("retrieval_context", "")
    
    expected_output = example.outputs.get("expected_output", "")
    
    if not retrieval_context:
        return {"key": "contextual_precision", "score": None, "comment": "Skipped: No RAG context retrieved for this query"}
        
    test_case = LLMTestCase(
        input=question,
        actual_output=response,
        expected_output=expected_output,
        retrieval_context=[retrieval_context],
    )
    metric = ContextualPrecisionMetric(threshold=0.70)
    try:
        metric.measure(test_case)
        return {"key": "contextual_precision", "score": metric.score, "comment": metric.reason}
    except Exception as e:
        return {"key": "contextual_precision", "score": 0, "comment": str(e)}

def contextual_recall_evaluator(run, example) -> dict:
    """Evaluate Contextual Recall using DeepEval."""
    inputs = example.inputs
    question = inputs.get("question") or inputs.get("input") or list(inputs.values())[0]
    
    outputs = run.outputs or {}
    response = outputs.get("response", "")
    state = outputs.get("state", {})
    retrieval_context = state.get("retrieval_context", "")
    
    expected_output = example.outputs.get("expected_output", "")
    
    if not retrieval_context:
        return {"key": "contextual_recall", "score": None, "comment": "Skipped: No RAG context retrieved for this query"}
        
    test_case = LLMTestCase(
        input=question,
        actual_output=response,
        expected_output=expected_output,
        retrieval_context=[retrieval_context],
    )
    metric = ContextualRecallMetric(threshold=0.70)
    try:
        metric.measure(test_case)
        return {"key": "contextual_recall", "score": metric.score, "comment": metric.reason}
    except Exception as e:
        return {"key": "contextual_recall", "score": 0, "comment": str(e)}

if __name__ == "__main__":
    dataset_name = "Life-Insurance-AI-Copilot"
    
    print(f"Starting LangSmith evaluation on dataset: {dataset_name}...")
    
    # Run the evaluation
    experiment_results = evaluate(
        predict_copilot,
        data=dataset_name,
        evaluators=[
            faithfulness_evaluator,
            relevancy_evaluator,
            relevance_evaluator,
            contextual_precision_evaluator,
            contextual_recall_evaluator,
        ],
        experiment_prefix="deepeval-metrics",
    )
    
    print("\nEvaluation complete! Check your LangSmith dashboard to view the experiment.")
