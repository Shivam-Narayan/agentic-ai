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
import logging
import sys
import time
import types
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LangSmith tracing setup
# ---------------------------------------------------------------------------
import os
_LANGSMITH_ENABLED = os.getenv("LANGCHAIN_TRACING_V2", "").lower() in ("true", "1", "yes")
if _LANGSMITH_ENABLED:
    print(f"✅ LangSmith tracing enabled (project: {os.getenv('LANGCHAIN_PROJECT', 'default')})")
    print(f"   View traces at: https://smith.langchain.com/")
else:
    print("⚠️  LangSmith tracing disabled. Set LANGCHAIN_TRACING_V2=true in .env to enable.")

# Patch: ragas imports ChatVertexAI which was removed from langchain_community
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _stub = types.ModuleType("langchain_community.chat_models.vertexai")
    _stub.ChatVertexAI = None
    sys.modules["langchain_community.chat_models.vertexai"] = _stub

from datasets import Dataset
from ragas import evaluate as ragas_evaluate
from ragas.metrics.collections import faithfulness, answer_relevancy, context_precision, context_recall
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
    for metric in [faithfulness, answer_relevancy, context_precision, context_recall]:
        metric.llm = llm
    answer_relevancy.embeddings = emb
    answer_relevancy.strictness = 1  # Groq caps n=1; strictness=1 uses single generation

METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]

# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------

async def _run_agent(question: str) -> tuple:
    """Returns (answer, contexts, datasource, latency, usage_metrics)."""
    from src.agent.workflow import aask
    from src.retrieval.rag import retrieve_documents
    t0     = time.monotonic()
    result = await aask(question, session_id=f"eval_{int(time.time())}")
    answer     = result.get("generation", "")
    datasource = result.get("datasource", "unknown")
    contexts   = []
    if datasource in ("company_docs", "multiple"):
        contexts = [d.page_content for d in retrieve_documents(question)]
    
    # Extract token usage metrics from the result
    usage_tracker = result.get("usage_tracker")
    usage_metrics = usage_tracker.to_metrics() if usage_tracker else None
    
    return answer, contexts, datasource, time.monotonic() - t0, usage_metrics

# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

async def evaluate(samples: list, verbose: bool) -> list:
    _setup_ragas()
    results = []
    total_cost = 0.0
    total_tokens = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for s in samples:
        q, gt = s["question"], s.get("ground_truth", "")
        print(f"\n[{s['id']}] {q}")

        try:
            answer, contexts, datasource, latency, usage_metrics = await _run_agent(q)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            results.append({**s, "error": str(exc)})
            continue

        # Extract cost and token metrics
        cost_usd = usage_metrics.cost_usd if usage_metrics else 0.0
        prompt_tokens = usage_metrics.prompt_tokens if usage_metrics else 0
        completion_tokens = usage_metrics.completion_tokens if usage_metrics else 0
        tokens = usage_metrics.total_tokens if usage_metrics else 0
        model = usage_metrics.model if usage_metrics else "unknown"
        
        total_cost += cost_usd
        total_tokens += tokens
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens

        has_ctx = bool(contexts)
        ctx     = contexts if has_ctx else ["(no context)"]

        dataset = Dataset.from_dict({
            "question":     [q],
            "contexts":     [ctx],
            "answer":       [answer],
            "ground_truth": [gt],
        })

        # Use all 4 metrics if context was retrieved, else only answer_relevancy
        metrics = METRICS if has_ctx else [answer_relevancy]
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
        print(f"     tokens: {tokens} ({prompt_tokens} prompt + {completion_tokens} completion) | cost: ${cost_usd:.6f} | model: {model}")

        if verbose:
            print(f"  answer  : {answer[:120]}")
            print(f"  expected: {gt[:120]}")

        results.append({
            **s, 
            "answer": answer, 
            "datasource": datasource,
            "faithfulness": faith, 
            "answer_relevancy": rel,
            "context_precision": prec, 
            "context_recall": rec, 
            "avg": avg,
            "latency_sec": round(latency, 2),
            "cost_usd": cost_usd,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": tokens,
            "model": model,
        })

    # Store aggregate metrics for summary
    results.append({
        "_aggregate": True,
        "total_cost_usd": round(total_cost, 6),
        "total_tokens": total_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "avg_cost_per_question": round(total_cost / len(results) if results else 0, 6),
        "avg_tokens_per_question": round(total_tokens / len(results) if results else 0, 2),
    })

    return results

# ---------------------------------------------------------------------------
# Regression comparison
# ---------------------------------------------------------------------------

def _load_baseline(baseline_path: Path) -> dict | None:
    """Load baseline evaluation results for comparison."""
    if not baseline_path.exists():
        return None
    try:
        return json.loads(baseline_path.read_text())
    except Exception as e:
        logger.warning(f"Failed to load baseline from {baseline_path}: {e}")
        return None


def _compare_with_baseline(current_results: list, baseline_data: dict | None) -> dict:
    """Compare current results with baseline and return regression analysis."""
    if not baseline_data:
        return {"has_baseline": False}
    
    baseline_results = baseline_data.get("results", [])
    baseline_aggregate = next((r for r in baseline_results if r.get("_aggregate")), None)
    current_aggregate = next((r for r in current_results if r.get("_aggregate")), None)
    
    if not baseline_aggregate or not current_aggregate:
        return {"has_baseline": False}
    
    baseline_scored = [r for r in baseline_results if "avg" in r and not r.get("_aggregate")]
    current_scored = [r for r in current_results if "avg" in r and not r.get("_aggregate")]
    
    def _avg_metric(results, key):
        vals = [r[key] for r in results if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else 0
    
    comparison = {
        "has_baseline": True,
        "baseline_date": baseline_data.get("timestamp", "unknown"),
        "metrics": {},
        "cost": {},
        "tokens": {},
    }
    
    # Compare RAGAS metrics
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        baseline_val = _avg_metric(baseline_scored, metric)
        current_val = _avg_metric(current_scored, metric)
        delta = current_val - baseline_val if baseline_val > 0 else 0
        comparison["metrics"][metric] = {
            "baseline": round(baseline_val, 4),
            "current": round(current_val, 4),
            "delta": round(delta, 4),
            "improved": delta > 0.01,  # Threshold: 1% improvement
            "regressed": delta < -0.01,  # Threshold: 1% regression
        }
    
    # Compare cost
    baseline_cost = baseline_aggregate.get("total_cost_usd", 0)
    current_cost = current_aggregate.get("total_cost_usd", 0)
    cost_delta = current_cost - baseline_cost
    comparison["cost"] = {
        "baseline": baseline_cost,
        "current": current_cost,
        "delta": round(cost_delta, 6),
        "delta_percent": round((cost_delta / baseline_cost * 100) if baseline_cost > 0 else 0, 2),
    }
    
    # Compare tokens
    baseline_tokens = baseline_aggregate.get("total_tokens", 0)
    current_tokens = current_aggregate.get("total_tokens", 0)
    token_delta = current_tokens - baseline_tokens
    comparison["tokens"] = {
        "baseline": baseline_tokens,
        "current": current_tokens,
        "delta": token_delta,
        "delta_percent": round((token_delta / baseline_tokens * 100) if baseline_tokens > 0 else 0, 2),
    }
    
    return comparison


def _print_comparison(comparison: dict) -> None:
    """Print regression comparison report."""
    if not comparison.get("has_baseline"):
        print("\n⚠️  No baseline found. Run evaluation with --save-baseline to create one.")
        return
    
    print(f"\n{'='*70}")
    print(f"  REGRESSION COMPARISON (baseline: {comparison['baseline_date']})")
    print(f"{'='*70}")
    
    print("\n  RAGAS Metrics:")
    for metric, data in comparison["metrics"].items():
        symbol = "📈" if data["improved"] else "📉" if data["regressed"] else "➡️"
        delta_str = f"{data['delta']:+.4f}"
        print(f"    {symbol} {metric:18s}: {data['current']:.4f} (baseline: {data['baseline']:.4f}, Δ {delta_str})")
    
    print("\n  Cost:")
    cost = comparison["cost"]
    cost_symbol = "💰" if cost["delta"] > 0 else "💚" if cost["delta"] < 0 else "➡️"
    print(f"    {cost_symbol} Total cost: ${cost['current']:.6f} (baseline: ${cost['baseline']:.6f}, Δ ${cost['delta']:+.6f} / {cost['delta_percent']:+.1f}%)")
    
    print("\n  Tokens:")
    tokens = comparison["tokens"]
    token_symbol = "⬆️" if tokens["delta"] > 0 else "⬇️" if tokens["delta"] < 0 else "➡️"
    print(f"    {token_symbol} Total tokens: {tokens['current']:,} (baseline: {tokens['baseline']:,}, Δ {tokens['delta']:+,} / {tokens['delta_percent']:+.1f}%)")
    
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(results: list) -> None:
    # Separate aggregate metrics from question results
    aggregate = next((r for r in results if r.get("_aggregate")), None)
    scored = [r for r in results if "avg" in r and not r.get("_aggregate")]
    
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
    
    print(f"\n{'='*70}")
    print(f"  RAGAS EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"  Questions        : {len(scored)} evaluated, {passed} passed  (threshold {PASS_THRESHOLD})")
    print(f"  faithfulness     : {_lbl(_avg('faithfulness'))}")
    print(f"  answer_relevancy : {_lbl(_avg('answer_relevancy'))}")
    print(f"  context_precision: {_lbl(_avg('context_precision'))}")
    print(f"  context_recall   : {_lbl(_avg('context_recall'))}")
    
    if aggregate:
        print(f"\n  COST & TOKEN METRICS")
        print(f"  {'─'*66}")
        print(f"  Total cost       : ${aggregate['total_cost_usd']:.6f}")
        print(f"  Avg cost/question: ${aggregate['avg_cost_per_question']:.6f}")
        print(f"  Total tokens     : {aggregate['total_tokens']:,} ({aggregate['total_prompt_tokens']:,} prompt + {aggregate['total_completion_tokens']:,} completion)")
        print(f"  Avg tokens/quest : {aggregate['avg_tokens_per_question']:.1f}")
        
        avg_latency = _avg('latency_sec')
        if avg_latency:
            print(f"  Avg latency      : {avg_latency:.2f}s")
    
    print(f"{'='*70}")
    print("\n✅ Check your LangSmith dashboard for detailed traces:")
    print("   https://smith.langchain.com/")
    print("   Filter by project: LANGCHAIN_PROJECT in your .env")

# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def test_rag_pipeline():
    """pytest entry point."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    samples = json.loads((Path(__file__).parent / "eval_questions.json").read_text())
    results = asyncio.run(evaluate(samples, verbose=False))
    print_report(results)
    scored  = [r for r in results if "avg" in r]
    assert scored, "No samples were scored"
    assert all(r["avg"] >= PASS_THRESHOLD for r in scored), \
        f"Failed: {[(r['id'], round(r['avg'],2)) for r in scored if r['avg'] < PASS_THRESHOLD]}"


def main():
    parser = argparse.ArgumentParser(description="Evaluate DataDialogue RAG pipeline.")
    parser.add_argument("--datasource", default=None,
                        choices=["company_docs", "database", "calculation", "direct_llm", "web_search"])
    parser.add_argument("--output",  default=None, metavar="PATH", help="Save results to JSON file")
    parser.add_argument("--save-baseline", action="store_true", help="Save results as baseline for future comparisons")
    parser.add_argument("--compare", default=None, metavar="PATH", help="Compare results against baseline file")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    samples = json.loads((Path(__file__).parent / "eval_questions.json").read_text())
    if args.datasource:
        samples = [s for s in samples if s.get("datasource") == args.datasource]

    results = asyncio.run(evaluate(samples, verbose=args.verbose))
    print_report(results)

    # Baseline comparison
    baseline_path = None
    if args.compare:
        baseline_path = Path(args.compare)
    elif args.save_baseline:
        baseline_path = Path(__file__).parent / "eval_baseline.json"
    else:
        # Auto-load baseline if it exists
        default_baseline = Path(__file__).parent / "eval_baseline.json"
        if default_baseline.exists():
            baseline_path = default_baseline
    
    if baseline_path and baseline_path.exists():
        baseline_data = _load_baseline(baseline_path)
        comparison = _compare_with_baseline(results, baseline_data)
        _print_comparison(comparison)

    # Save results
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "datasource_filter": args.datasource,
        "results": results,
    }

    if args.output:
        Path(args.output).write_text(json.dumps(output_data, indent=2))
        print(f"💾 Results saved to {args.output}")

    if args.save_baseline:
        baseline_file = Path(__file__).parent / "eval_baseline.json"
        baseline_file.write_text(json.dumps(output_data, indent=2))
        print(f"📊 Baseline saved to {baseline_file}")
        print("   Use --compare eval_baseline.json to compare future runs")


if __name__ == "__main__":
    main()
