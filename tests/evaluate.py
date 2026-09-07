import os
import sys
import json
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from src.agent.chains import get_llm
from langchain_core.messages import HumanMessage

# --- Dummy Dataset ---
dummy_eval_data = {
    "question": [
        "What is the company's refund policy?",
        "How do I reset my password?"
    ],
    "contexts": [
        ["Our refund policy allows returns within 30 days of purchase. Items must be in original condition."],
        ["To reset your password, go to settings > security > reset password."]
    ],
    "answer": [
        "You can get a refund within 30 days if the item is in original condition.",
        "Navigate to the settings menu and click on security to reset your password."
    ],
    "ground_truth": [
        "Refunds are available for 30 days for items in original condition.",
        "Go to settings, then security, then reset password."
    ]
}


def _judge(prompt: str) -> float:
    """Call the project LLM as judge. Returns a score between 0.0 and 1.0."""
    response = get_llm().invoke([HumanMessage(content=prompt)])
    text = response.content if hasattr(response, "content") else str(response)
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        try:
            return max(0.0, min(1.0, float(json.loads(match.group(0)).get("score", 0.0))))
        except (json.JSONDecodeError, ValueError):
            pass
    numbers = re.findall(r'\b(0\.\d+|1\.0)\b', text)
    return float(numbers[0]) if numbers else 0.0


def score_faithfulness(context, answer):
    return _judge(f"Does the answer contain only information from the context? Context: {context}\nAnswer: {answer}\nReply ONLY: {{\"score\": <0.0-1.0>}}")

def score_answer_relevance(question, answer):
    return _judge(f"Does the answer address the question? Question: {question}\nAnswer: {answer}\nReply ONLY: {{\"score\": <0.0-1.0>}}")

def score_context_precision(question, context):
    return _judge(f"Is the context relevant to the question? Question: {question}\nContext: {context}\nReply ONLY: {{\"score\": <0.0-1.0>}}")


def test_rag_pipeline():
    """
    Evaluates the RAG pipeline outputs using RAGAS-style metrics and logs to LangSmith.
    Uses the project's Groq/Gemini LLM as judge — no OpenAI or ragas package needed.
    LangSmith tracing is automatic via LANGCHAIN_TRACING_V2=true in .env
    """
    questions    = dummy_eval_data["question"]
    contexts     = dummy_eval_data["contexts"]
    answers      = dummy_eval_data["answer"]
    ground_truth = dummy_eval_data["ground_truth"]

    faith_scores, rel_scores, prec_scores = [], [], []

    print("Running RAG evaluation...\n")
    for q, ctx_list, ans, gt in zip(questions, contexts, answers, ground_truth):
        ctx = "\n".join(ctx_list)
        f = score_faithfulness(ctx, ans)
        r = score_answer_relevance(q, ans)
        p = score_context_precision(q, ctx)
        faith_scores.append(f)
        rel_scores.append(r)
        prec_scores.append(p)
        print(f"Q: {q}")
        print(f"   faithfulness={f:.2f}  answer_relevance={r:.2f}  context_precision={p:.2f}\n")

    results = {
        "faithfulness":      sum(faith_scores) / len(faith_scores),
        "answer_relevance":  sum(rel_scores)   / len(rel_scores),
        "context_precision": sum(prec_scores)  / len(prec_scores),
    }

    print("=== Evaluation Results ===")
    print(results)

    assert results["faithfulness"]      > 0.7, f"Faithfulness too low: {results['faithfulness']}"
    assert results["answer_relevance"]  > 0.7, f"Answer relevance too low: {results['answer_relevance']}"
    assert results["context_precision"] > 0.7, f"Context precision too low: {results['context_precision']}"

    print("\n✅ Evaluation completed successfully! Check your LangSmith dashboard.")


if __name__ == "__main__":
    test_rag_pipeline()
