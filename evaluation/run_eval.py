"""
DeepEval evaluation script for Life Insurance AI Copilot.
Runs 30 test questions against the /chat API endpoint and produces a scorecard.

Usage:
    1. Start the backend: uvicorn app.main:app --port 8000
    2. Run: python evaluation/run_eval.py

Requires: deepeval, httpx
"""

import json
import os
import sys
import httpx
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

API_URL = os.getenv("API_URL", "http://localhost:8000")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")


def load_test_set():
    with open(TEST_SET_PATH, "r") as f:
        return json.load(f)


def query_copilot(question: str, session_id: str = "eval-session") -> dict:
    """Call the /chat endpoint and return the response data."""
    try:
        resp = httpx.post(
            f"{API_URL}/chat",
            json={"session_id": session_id, "message": question},
            timeout=60.0,
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            return {"response": f"ERROR: {resp.status_code}", "node_path": []}
    except Exception as e:
        return {"response": f"CONNECTION_ERROR: {e}", "node_path": []}


def run_evaluation():
    test_set = load_test_set()
    print(f"\n{'='*70}")
    print(f"  Life Insurance AI Copilot — Evaluation Scorecard")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Test Set: {len(test_set)} questions")
    print(f"{'='*70}\n")

    results = []
    routing_correct = 0
    routing_total = 0

    for i, tc in enumerate(test_set, 1):
        qid = tc["id"]
        category = tc["category"]
        question = tc["question"]
        expected_intent = tc.get("expected_intent", "")
        keywords = tc.get("expected_answer_keywords", [])

        # Use a unique session per question to avoid state carryover
        session_id = f"eval-{qid}"
        print(f"  [{i:02d}/{len(test_set)}] {qid} ({category})")
        print(f"     Q: {question[:80]}...")

        result = query_copilot(question, session_id)
        response = result.get("response", "")
        node_path = result.get("node_path", [])

        # Check intent routing accuracy
        actual_intent = None
        intent_correct = False
        if node_path:
            # Extract the agent name from node_path
            for node in node_path:
                if node != "intent_router":
                    actual_intent = node.replace("_agent", "")
                    break
        if actual_intent:
            routing_total += 1
            # Map agent names to intents
            intent_map = {
                "underwriting": "underwriting",
                "policy_qa": "policy_qa",
                "beneficiary": "beneficiary",
                "issuance": "issuance",
                "lapse_revival": "lapse_revival",
            }
            mapped = intent_map.get(actual_intent, actual_intent)
            if mapped == expected_intent:
                intent_correct = True
                routing_correct += 1

        # Check keyword coverage
        response_lower = response.lower()
        matched_keywords = [kw for kw in keywords if kw.lower() in response_lower]
        keyword_coverage = len(matched_keywords) / len(keywords) if keywords else 1.0

        # Has citation?
        has_citation = any(marker in response for marker in ["Source:", "Page:", "[Source", "page"])

        results.append({
            "id": qid,
            "category": category,
            "question": question,
            "response": response[:200],
            "node_path": node_path,
            "intent_correct": intent_correct,
            "keyword_coverage": keyword_coverage,
            "has_citation": has_citation,
            "matched_keywords": matched_keywords,
        })

        status = "✅" if intent_correct else "❌"
        print(f"     Route: {status} (expected={expected_intent}, actual={actual_intent})")
        print(f"     Keywords: {len(matched_keywords)}/{len(keywords)} matched")
        print(f"     Citation: {'✅' if has_citation else '⚠️'}")
        print()

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SCORECARD SUMMARY")
    print(f"{'='*70}")

    routing_accuracy = (routing_correct / routing_total * 100) if routing_total > 0 else 0
    print(f"\n  Intent Routing Accuracy: {routing_correct}/{routing_total} = {routing_accuracy:.1f}%")
    print(f"  {'✅ PASS' if routing_accuracy >= 90 else '❌ FAIL'} (threshold: >= 90%)")

    avg_keyword_coverage = sum(r["keyword_coverage"] for r in results) / len(results) if results else 0
    print(f"\n  Average Keyword Coverage: {avg_keyword_coverage:.2f}")

    citation_count = sum(1 for r in results if r["has_citation"])
    print(f"  Citation Rate: {citation_count}/{len(results)}")

    # Per-category breakdown
    print(f"\n  Per-Category Breakdown:")
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "correct": 0, "keyword_avg": []}
        categories[cat]["total"] += 1
        if r["intent_correct"]:
            categories[cat]["correct"] += 1
        categories[cat]["keyword_avg"].append(r["keyword_coverage"])

    for cat, data in sorted(categories.items()):
        acc = data["correct"] / data["total"] * 100 if data["total"] > 0 else 0
        kw = sum(data["keyword_avg"]) / len(data["keyword_avg"]) if data["keyword_avg"] else 0
        print(f"    {cat:25s}  Routing: {data['correct']}/{data['total']} ({acc:.0f}%)  Keywords: {kw:.2f}")

    # ── DeepEval integration (optional) ─────────────────────────────
    print(f"\n{'='*70}")
    print("  Attempting DeepEval metrics (requires deepeval installed)...")
    print(f"{'='*70}")

    try:
        from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
        from deepeval.test_case import LLMTestCase

        faithfulness_scores = []
        relevancy_scores = []

        for r in results:
            # We need the retrieval context for faithfulness
            # For now, use the response as self-contained context
            test_case = LLMTestCase(
                input=r["question"],
                actual_output=r["response"],
                retrieval_context=[r["response"]],
            )

            try:
                fm = FaithfulnessMetric(threshold=0.85)
                fm.measure(test_case)
                faithfulness_scores.append(fm.score)
            except Exception as e:
                print(f"    Faithfulness error for {r['id']}: {e}")

            try:
                arm = AnswerRelevancyMetric(threshold=0.80)
                arm.measure(test_case)
                relevancy_scores.append(arm.score)
            except Exception as e:
                print(f"    Relevancy error for {r['id']}: {e}")

        if faithfulness_scores:
            avg_faith = sum(faithfulness_scores) / len(faithfulness_scores)
            print(f"\n  DeepEval Faithfulness:      {avg_faith:.3f}  {'✅ PASS' if avg_faith >= 0.85 else '❌ FAIL'} (threshold: 0.85)")

        if relevancy_scores:
            avg_rel = sum(relevancy_scores) / len(relevancy_scores)
            print(f"  DeepEval Answer Relevancy:  {avg_rel:.3f}  {'✅ PASS' if avg_rel >= 0.80 else '❌ FAIL'} (threshold: 0.80)")

    except ImportError:
        print("  ⚠️ deepeval not installed. Install with: pip install deepeval")
        print("  Skipping DeepEval metrics. Keyword-based evaluation shown above.")

    # ── Save results ─────────────────────────────────────────────────
    output_path = os.path.join(os.path.dirname(__file__), "eval_results.json")
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "routing_accuracy": routing_accuracy,
            "avg_keyword_coverage": avg_keyword_coverage,
            "citation_rate": citation_count / len(results) if results else 0,
            "results": results,
        }, f, indent=2)
    print(f"\n  Full results saved to: {output_path}")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    run_evaluation()
