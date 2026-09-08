# -*- coding: utf-8 -*-
"""
DataDialogue — RAG Evaluation
Evaluates the live agent using real Ragas metrics (no OpenAI needed).
Groq/Gemini LLM + HuggingFace embeddings are injected as the judge.

Usage:
  python tests/evaluate.py
  python tests/evaluate.py --datasource company_docs
  python tests/evaluate.py --output results.json
"""

import argparse
import asyncio
import json
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# Patch: ragas imports ChatVertexAI which was removed from langchain_community
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _stub = types.ModuleType("langchain_community.chat_models.vertexai")
    _stub.ChatVertexAI = None
    sys.modules["langchain_community.chat_models.vertexai"] = _stub

from datasets import Dataset
from ragas import evaluate as ragas_evaluate
from ragas.metrics import _faithfulness, _answer_relevancy, _context_precision, _context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.agent.chains import get_llm

PASS_THRESHOLD = 0.70

# ---------------------------------------------------------------------------
# Inject project LLM + embeddings into ragas (replaces OpenAI)
# ---------------------------------------------------------------------------

def _setup_ragas():
    llm = LangchainLLMWrapper(get_llm())
    emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5"))
    for metric in [_faithfulness, _answer_relevancy, _context_precision, _context_recall]:
        metric.llm = llm
    _answer_relevancy.embeddings = emb
    _answer_relevancy.strictness = 1  # Groq caps n=1; strictness=1 uses single generation

METRICS = [_faithfulness, _answer_relevancy, _context_precision, _context_recall]

# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------

async def _run_agent(question: str) -> tuple:
    """Returns (answer, contexts, datasource, latency)."""
    from src.agent.workflow import aask
    from src.agent.rag import retrieve_documents
    t0     = time.monotonic()
    result = await aask(question, session_id=f"eval_{int(time.time())}")
    answer     = result.get("generation", "")
    datasource = result.get("datasource", "unknown")
    contexts   = []
    if datasource in ("company_docs", "multiple"):
        contexts = [d.page_content for d in retrieve_documents(question)]
    return answer, contexts, datasource, time.monotonic() - t0

# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(samples: list, verbose: bool) -> list:
    _setup_ragas()
    results = []

    for s in samples:
        q, gt = s["question"], s.get("ground_truth", "")
        print(f"\n[{s['id']}] {q}")

        try:
            answer, contexts, datasource, latency = asyncio.run(_run_agent(q))
        except Exception as exc:
            print(f"  ERROR: {exc}")
            results.append({**s, "error": str(exc)})
            continue

        has_ctx = bool(contexts)
        ctx     = contexts if has_ctx else ["(no context)"]

        dataset = Dataset.from_dict({
            "question":     [q],
            "contexts":     [ctx],
            "answer":       [answer],
            "ground_truth": [gt],
        })

        # Use all 4 metrics if context was retrieved, else only answer_relevancy
        metrics = METRICS if has_ctx else [_answer_relevancy]
        scores  = ragas_evaluate(dataset, metrics=metrics)

        faith = float(scores["faithfulness"][0])      if has_ctx else None
        rel   = float(scores["answer_relevancy"][0])
        prec  = float(scores["context_precision"][0]) if has_ctx else None
        rec   = float(scores["context_recall"][0])    if has_ctx else None

        active = [v for v in [faith, rel, prec, rec] if v is not None]
        avg    = sum(active) / len(active)
        flag   = "✅" if avg >= PASS_THRESHOLD else "❌"

        f_s = f"{faith:.2f}" if faith is not None else "n/a"
        p_s = f"{prec:.2f}"  if prec  is not None else "n/a"
        r_s = f"{rec:.2f}"   if rec   is not None else "n/a"
        print(f"  {flag} avg={avg:.2f}  faith={f_s}  rel={rel:.2f}  prec={p_s}  recall={r_s}  ({latency:.1f}s)")

        if verbose:
            print(f"  answer  : {answer[:120]}")
            print(f"  expected: {gt[:120]}")

        results.append({**s, "answer": answer, "datasource": datasource,
                        "faithfulness": faith, "answer_relevancy": rel,
                        "context_precision": prec, "context_recall": rec, "avg": avg})

    return results

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(results: list) -> None:
    scored = [r for r in results if "avg" in r]
    if not scored:
        print("\nNo scored results.")
        return

    def _avg(key):
        vals = [r[key] for r in scored if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    def _lbl(v):
        if v is None: return "n/a"
        return f"{v:.4f} {'✅' if v >= PASS_THRESHOLD else '❌'}"

    passed = sum(1 for r in scored if r["avg"] >= PASS_THRESHOLD)
    print(f"\n{'='*55}")
    print(f"  Results          : {passed}/{len(scored)} passed  (threshold {PASS_THRESHOLD})")
    print(f"  faithfulness     : {_lbl(_avg('faithfulness'))}")
    print(f"  answer_relevancy : {_lbl(_avg('answer_relevancy'))}")
    print(f"  context_precision: {_lbl(_avg('context_precision'))}")
    print(f"  context_recall   : {_lbl(_avg('context_recall'))}")
    print(f"{'='*55}")
    print("\n✅ Check your LangSmith dashboard for detailed traces.")

# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def test_rag_pipeline():
    """pytest entry point."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    samples = json.loads((Path(__file__).parent / "eval_questions.json").read_text())
    results = evaluate(samples, verbose=False)
    print_report(results)
    scored  = [r for r in results if "avg" in r]
    assert scored, "No samples were scored"
    assert all(r["avg"] >= PASS_THRESHOLD for r in scored), \
        f"Failed: {[(r['id'], round(r['avg'],2)) for r in scored if r['avg'] < PASS_THRESHOLD]}"


def main():
    parser = argparse.ArgumentParser(description="Evaluate DataDialogue RAG pipeline.")
    parser.add_argument("--datasource", default=None,
                        choices=["company_docs", "database", "calculation", "direct_llm", "web_search"])
    parser.add_argument("--output",  default=None, metavar="PATH")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    samples = json.loads((Path(__file__).parent / "eval_questions.json").read_text())
    if args.datasource:
        samples = [s for s in samples if s.get("datasource") == args.datasource]

    results = evaluate(samples, verbose=args.verbose)
    print_report(results)

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
