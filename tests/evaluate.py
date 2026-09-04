# -*- coding: utf-8 -*-
"""
DataDialogue — RAG Evaluation Script
=====================================

Evaluates the agent's answer quality using LLM-as-judge methodology,
measuring the same four metrics as RAGAS:

  faithfulness       — answer is grounded in retrieved context (no hallucination)
  answer_relevancy   — answer actually addresses the question
  context_precision  — retrieved chunks are relevant to the question
  context_recall     — all relevant information was retrieved

Usage
-----
  # Run all questions
  python tests/evaluate.py

  # Filter by datasource
  python tests/evaluate.py --datasource company_docs
  python tests/evaluate.py --datasource database
  python tests/evaluate.py --datasource calculation
  python tests/evaluate.py --datasource direct_llm

  # Save results to JSON
  python tests/evaluate.py --output results.json

  # Verbose mode (show per-question details)
  python tests/evaluate.py --verbose

Notes
-----
  - FastAPI does NOT need to be running — calls aask() directly.
  - Uses your existing Groq LLM as the judge — no new API keys.
  - Does not modify any existing application code.
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — allow imports from project root
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Logging — quiet down noisy third-party libraries during evaluation
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("evaluate")

# Silence LlamaIndex weight loading progress bars
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("llama_index").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Score thresholds — used to colour-code results in the report
# ---------------------------------------------------------------------------

SCORE_EXCELLENT: float = 0.85
SCORE_GOOD:      float = 0.70
SCORE_POOR:      float = 0.50


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EvalQuestion:
    """One test question with its expected answer."""
    id:           str
    question:     str
    ground_truth: str
    datasource:   str
    notes:        str = ""


@dataclass
class EvalResult:
    """Evaluation result for a single question."""
    id:                  str
    question:            str
    ground_truth:        str
    agent_answer:        str
    retrieved_contexts:  list[str]
    expected_datasource: str
    actual_datasource:   str
    tools_used:          list[str]

    # Metric scores (0.0 — 1.0)
    faithfulness:       float = 0.0
    answer_relevancy:   float = 0.0
    context_precision:  float = 0.0
    context_recall:     float = 0.0

    # Meta
    latency_secs:  float = 0.0
    error:         str   = ""
    skipped:       bool  = False


@dataclass
class EvalReport:
    """Aggregated evaluation report."""
    total:              int   = 0
    passed:             int   = 0
    failed:             int   = 0
    skipped:            int   = 0
    avg_faithfulness:   float = 0.0
    avg_relevancy:      float = 0.0
    avg_precision:      float = 0.0
    avg_recall:         float = 0.0
    avg_latency_secs:   float = 0.0
    results:            list[EvalResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# LLM judge prompts
# ---------------------------------------------------------------------------

_FAITHFULNESS_PROMPT = """\
You are an objective evaluator. Given a context and an answer, determine if the
answer is fully supported by the context (no hallucination).

Context:
{context}

Answer:
{answer}

Score the faithfulness on a scale of 0.0 to 1.0:
- 1.0: Every claim in the answer is directly supported by the context
- 0.7: Most claims are supported; minor extrapolation present
- 0.5: Some claims supported; some not found in context
- 0.0: Answer contradicts or ignores the context entirely

Reply with ONLY a JSON object: {{"score": <float>, "reason": "<one sentence>"}}"""


_RELEVANCY_PROMPT = """\
You are an objective evaluator. Given a question and an answer, determine how
well the answer addresses the question.

Question:
{question}

Answer:
{answer}

Score the answer relevancy on a scale of 0.0 to 1.0:
- 1.0: Answer directly and completely addresses the question
- 0.7: Answer mostly addresses the question with minor gaps
- 0.5: Answer partially addresses the question
- 0.0: Answer does not address the question at all

Reply with ONLY a JSON object: {{"score": <float>, "reason": "<one sentence>"}}"""


_PRECISION_PROMPT = """\
You are an objective evaluator. Given a question and retrieved context chunks,
determine how relevant the retrieved chunks are to answering the question.

Question:
{question}

Retrieved context:
{context}

Score the context precision on a scale of 0.0 to 1.0:
- 1.0: All retrieved chunks are highly relevant to the question
- 0.7: Most chunks are relevant; a few are off-topic
- 0.5: About half the chunks are relevant
- 0.0: Retrieved chunks are completely irrelevant to the question

Reply with ONLY a JSON object: {{"score": <float>, "reason": "<one sentence>"}}"""


_RECALL_PROMPT = """\
You are an objective evaluator. Given a question, the correct answer, and
retrieved context chunks, determine if the context contains all the information
needed to answer the question correctly.

Question:
{question}

Correct answer:
{ground_truth}

Retrieved context:
{context}

Score the context recall on a scale of 0.0 to 1.0:
- 1.0: Context contains all information needed to derive the correct answer
- 0.7: Context contains most of the needed information
- 0.5: Context contains some relevant information but key facts are missing
- 0.0: Context does not contain the information needed to answer correctly

Reply with ONLY a JSON object: {{"score": <float>, "reason": "<one sentence>"}}"""


# ---------------------------------------------------------------------------
# Judge LLM
# ---------------------------------------------------------------------------

def _get_judge_llm():
    """Return the project's configured LLM for use as an evaluation judge.

    Reuses get_llm() from chains.py — no new API keys required.
    The LLM is called without tools (bare LLM call, no agent loop).
    """
    from src.agent.chains import get_llm
    return get_llm()


def _parse_score_response(response_text: str, metric: str) -> tuple[float, str]:
    """Parse a score+reason JSON from the judge LLM response.

    Returns (score, reason). Falls back to 0.0 on parse failure.
    """
    text = response_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            l for l in lines
            if not l.startswith("```")
        ).strip()

    try:
        data   = json.loads(text)
        score  = float(data.get("score", 0.0))
        reason = str(data.get("reason", ""))
        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, score)), reason
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning("Could not parse %s score response: %s — raw: %r", metric, exc, text[:200])
        return 0.0, "parse error"


async def _judge_score(prompt: str, metric: str) -> tuple[float, str]:
    """Ask the judge LLM to score one metric. Returns (score, reason)."""
    from langchain_core.messages import HumanMessage
    llm = _get_judge_llm()
    try:
        response = await asyncio.wait_for(
            llm.ainvoke([HumanMessage(content=prompt)]),
            timeout=30,
        )
        return _parse_score_response(
            response.content if hasattr(response, "content") else str(response),
            metric,
        )
    except asyncio.TimeoutError:
        logger.warning("Judge LLM timed out scoring %s", metric)
        return 0.0, "timeout"
    except Exception as exc:
        logger.warning("Judge LLM error scoring %s: %s", metric, exc)
        return 0.0, str(exc)


# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------

async def _run_agent(question: str) -> tuple[str, list[str], str, list[str], float]:
    """Call aask() and return (answer, contexts, datasource, tools_used, latency).

    Extracts retrieved context chunks from the tool messages in the agent's
    message history so we can pass them to the judge.
    """
    from src.agent.workflow import aask
    from src.agent.rag import retrieve_documents

    t0 = time.monotonic()
    try:
        result = await aask(question, session_id=f"eval_{int(time.time())}")
    except Exception as exc:
        return "", [], "error", [], time.monotonic() - t0

    latency = time.monotonic() - t0

    answer     = result.get("generation", "")
    datasource = result.get("datasource", "unknown")
    tools_used = result.get("tools_used", [])

    # Retrieve the same chunks the agent would have used for context scoring
    # We re-retrieve here because the raw chunks aren't directly exposed
    # by aask() — only the formatted answer is.
    contexts: list[str] = []
    if datasource in ("company_docs", "multiple"):
        try:
            docs = retrieve_documents(question)
            contexts = [d.page_content for d in docs]
        except Exception as exc:
            logger.warning("Could not retrieve contexts for eval: %s", exc)

    return answer, contexts, datasource, tools_used, latency


# ---------------------------------------------------------------------------
# Per-question evaluator
# ---------------------------------------------------------------------------

async def _evaluate_question(q: EvalQuestion, verbose: bool = False) -> EvalResult:
    """Run the agent + judge for one question and return an EvalResult."""
    print(f"  [{q.id}] {q.question[:70]}...")

    result = EvalResult(
        id=q.id,
        question=q.question,
        ground_truth=q.ground_truth,
        agent_answer="",
        retrieved_contexts=[],
        expected_datasource=q.datasource,
        actual_datasource="",
        tools_used=[],
    )

    # Step 1 — get agent answer
    try:
        answer, contexts, datasource, tools_used, latency = await _run_agent(q.question)
    except Exception as exc:
        result.error   = str(exc)
        result.skipped = True
        print(f"         ERROR: {exc}")
        return result

    result.agent_answer        = answer
    result.retrieved_contexts  = contexts
    result.actual_datasource   = datasource
    result.tools_used          = tools_used
    result.latency_secs        = latency

    if not answer:
        result.error   = "Agent returned empty answer"
        result.skipped = True
        print("         SKIPPED: empty answer")
        return result

    context_text = "\n\n---\n\n".join(contexts) if contexts else "(no context retrieved)"

    # Step 2 — score all 4 metrics concurrently
    scores = await asyncio.gather(
        _judge_score(
            _FAITHFULNESS_PROMPT.format(context=context_text, answer=answer),
            "faithfulness",
        ),
        _judge_score(
            _RELEVANCY_PROMPT.format(question=q.question, answer=answer),
            "answer_relevancy",
        ),
        _judge_score(
            _PRECISION_PROMPT.format(question=q.question, context=context_text),
            "context_precision",
        ),
        _judge_score(
            _RECALL_PROMPT.format(
                question=q.question,
                ground_truth=q.ground_truth,
                context=context_text,
            ),
            "context_recall",
        ),
    )

    result.faithfulness,      faith_reason  = scores[0]
    result.answer_relevancy,  rel_reason    = scores[1]
    result.context_precision, prec_reason   = scores[2]
    result.context_recall,    recall_reason = scores[3]

    if verbose:
        print(f"         answer    : {answer[:100]}...")
        print(f"         datasource: {datasource} (expected: {q.datasource})")
        print(f"         latency   : {latency:.1f}s")
        print(f"         faith     : {result.faithfulness:.2f} — {faith_reason}")
        print(f"         relevancy : {result.answer_relevancy:.2f} — {rel_reason}")
        print(f"         precision : {result.context_precision:.2f} — {prec_reason}")
        print(f"         recall    : {result.context_recall:.2f} — {recall_reason}")
    else:
        avg = (result.faithfulness + result.answer_relevancy +
               result.context_precision + result.context_recall) / 4
        flag = "✅" if avg >= SCORE_GOOD else "⚠️" if avg >= SCORE_POOR else "❌"
        print(f"         {flag} avg={avg:.2f}  faith={result.faithfulness:.2f}  "
              f"rel={result.answer_relevancy:.2f}  "
              f"prec={result.context_precision:.2f}  "
              f"recall={result.context_recall:.2f}  "
              f"({latency:.1f}s)")

    return result


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def _build_report(results: list[EvalResult]) -> EvalReport:
    """Aggregate individual results into a summary report."""
    report = EvalReport(total=len(results))

    scored = [r for r in results if not r.skipped and not r.error]
    report.skipped = sum(1 for r in results if r.skipped or r.error)
    report.passed  = sum(
        1 for r in scored
        if (r.faithfulness + r.answer_relevancy) / 2 >= SCORE_GOOD
    )
    report.failed  = len(scored) - report.passed

    if scored:
        report.avg_faithfulness  = sum(r.faithfulness      for r in scored) / len(scored)
        report.avg_relevancy     = sum(r.answer_relevancy  for r in scored) / len(scored)
        report.avg_precision     = sum(r.context_precision for r in scored) / len(scored)
        report.avg_recall        = sum(r.context_recall    for r in scored) / len(scored)
        report.avg_latency_secs  = sum(r.latency_secs      for r in scored) / len(scored)

    report.results = results
    return report


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def _score_label(score: float) -> str:
    """Return a coloured label string for a score."""
    if score >= SCORE_EXCELLENT:
        return f"{score:.2f} ✅"
    if score >= SCORE_GOOD:
        return f"{score:.2f} ✅"
    if score >= SCORE_POOR:
        return f"{score:.2f} ⚠️"
    return f"{score:.2f} ❌"


def _improvement_hint(report: EvalReport) -> str:
    """Return the most actionable improvement hint based on scores."""
    scores = {
        "faithfulness"      : report.avg_faithfulness,
        "answer_relevancy"  : report.avg_relevancy,
        "context_precision" : report.avg_precision,
        "context_recall"    : report.avg_recall,
    }
    weakest = min(scores, key=scores.get)
    hints = {
        "faithfulness": (
            "faithfulness",
            "LLM may be hallucinating. Tighten the system prompt in prompt.py — "
            "add explicit instruction: 'Only use information from the provided context.'"
        ),
        "answer_relevancy": (
            "answer_relevancy",
            "LLM is going off-topic. Reduce recursion_limit in workflow.py "
            "and simplify the system prompt."
        ),
        "context_precision": (
            "context_precision",
            "Wrong chunks retrieved. Try reducing chunk_size from 512 to 256 "
            "in rag.py SentenceSplitter, or improve query formulation in tools.py."
        ),
        "context_recall": (
            "context_recall",
            "Missing relevant chunks. Increase similarity_top_k from 8 to 12 "
            "in rag.py retrieve_documents()."
        ),
    }
    metric, hint = hints[weakest]
    return f"Weakest: {metric} ({scores[weakest]:.2f})\n  Hint: {hint}"


def _print_report(report: EvalReport) -> None:
    """Print a formatted evaluation report to stdout."""
    w = 60
    print()
    print("=" * w)
    print("  DataDialogue — RAGAS Evaluation Report")
    print("=" * w)
    print(f"  Questions evaluated : {report.total}")
    print(f"  Passed              : {report.passed}")
    print(f"  Failed              : {report.failed}")
    print(f"  Skipped / Errored   : {report.skipped}")
    print(f"  Avg latency         : {report.avg_latency_secs:.1f}s")
    print("-" * w)
    print(f"  Faithfulness        : {_score_label(report.avg_faithfulness)}")
    print(f"  Answer relevancy    : {_score_label(report.avg_relevancy)}")
    print(f"  Context precision   : {_score_label(report.avg_precision)}")
    print(f"  Context recall      : {_score_label(report.avg_recall)}")
    print("-" * w)

    scored = [r for r in report.results if not r.skipped and not r.error]
    if scored:
        print()
        print(_improvement_hint(report))

    # Datasource breakdown
    datasources: dict[str, list[EvalResult]] = {}
    for r in scored:
        ds = r.expected_datasource
        datasources.setdefault(ds, []).append(r)

    if len(datasources) > 1:
        print()
        print("  By datasource:")
        for ds, ds_results in sorted(datasources.items()):
            avg_faith = sum(r.faithfulness     for r in ds_results) / len(ds_results)
            avg_rel   = sum(r.answer_relevancy for r in ds_results) / len(ds_results)
            avg       = (avg_faith + avg_rel) / 2
            flag      = "✅" if avg >= SCORE_GOOD else "⚠️" if avg >= SCORE_POOR else "❌"
            print(f"    {flag} {ds:<20} n={len(ds_results)}  "
                  f"faith={avg_faith:.2f}  rel={avg_rel:.2f}")

    # Failed questions detail
    failed = [r for r in scored if (r.faithfulness + r.answer_relevancy) / 2 < SCORE_GOOD]
    if failed:
        print()
        print("  Failed questions:")
        for r in failed:
            avg = (r.faithfulness + r.answer_relevancy) / 2
            print(f"    [{r.id}] {r.question[:55]}... (avg={avg:.2f})")
            print(f"           expected: {r.ground_truth[:80]}")
            print(f"           got     : {r.agent_answer[:80]}")

    print()
    print("=" * w)
    print()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _load_questions(datasource_filter: str | None) -> list[EvalQuestion]:
    """Load and optionally filter questions from eval_questions.json."""
    questions_path = Path(__file__).parent / "eval_questions.json"
    if not questions_path.exists():
        raise FileNotFoundError(
            f"Test questions file not found: {questions_path}\n"
            "Expected: tests/eval_questions.json"
        )

    raw = json.loads(questions_path.read_text(encoding="utf-8"))
    questions = [EvalQuestion(**q) for q in raw]

    if datasource_filter:
        questions = [q for q in questions if q.datasource == datasource_filter]
        if not questions:
            valid = sorted({q.datasource for q in [EvalQuestion(**r) for r in raw]})
            raise ValueError(
                f"No questions found for datasource '{datasource_filter}'. "
                f"Valid options: {valid}"
            )

    return questions


async def _run_evaluation(
    datasource_filter: str | None,
    output_path: str | None,
    verbose: bool,
) -> EvalReport:
    """Run the full evaluation pipeline and return the report."""
    questions = _load_questions(datasource_filter)

    filter_label = f" (datasource={datasource_filter})" if datasource_filter else ""
    print()
    print(f"DataDialogue RAG Evaluation{filter_label}")
    print(f"Running {len(questions)} question(s)...\n")

    results: list[EvalResult] = []
    for q in questions:
        result = await _evaluate_question(q, verbose=verbose)
        results.append(result)
        # Small delay to avoid hitting Groq rate limits between questions
        await asyncio.sleep(2)

    report = _build_report(results)
    _print_report(report)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        serialisable = asdict(report)
        out.write_text(json.dumps(serialisable, indent=2), encoding="utf-8")
        print(f"Results saved to: {output_path}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate DataDialogue agent answer quality using LLM-as-judge.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/evaluate.py
  python tests/evaluate.py --datasource company_docs
  python tests/evaluate.py --datasource database --verbose
  python tests/evaluate.py --output tests/results.json
        """,
    )
    parser.add_argument(
        "--datasource",
        choices=["company_docs", "database", "calculation", "direct_llm", "web_search"],
        default=None,
        help="Only evaluate questions for this datasource.",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Save full results as JSON to this path.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Show per-question answer and score breakdown.",
    )
    args = parser.parse_args()

    # Windows: psycopg3 async requires SelectorEventLoop
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(
        _run_evaluation(
            datasource_filter=args.datasource,
            output_path=args.output,
            verbose=args.verbose,
        )
    )


if __name__ == "__main__":
    main()
