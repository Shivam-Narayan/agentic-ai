import logging
from functools import lru_cache
from typing import List, Literal

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from tenacity import retry, stop_after_attempt, wait_exponential
from typing_extensions import TypedDict

from .chains import get_chains
from .rag import retrieve_documents

MAX_RETRIES = 3
logger = logging.getLogger(__name__)

Datasource = Literal["vectorstore", "websearch"]
GradeDecision = Literal["useful", "not useful", "not supported", "max_retries"]
GenerateOrSearch = Literal["generate", "websearch"]


# --------------------------------------------------
# 4. LangGraph State (global state)
# --------------------------------------------------

class AgentState(TypedDict, total=False):
    question: str
    retries: int
    datasource: Datasource
    documents: List[Document]
    generation: str
    web_search: Literal["Yes", "No"]
    error: str


# --------------------------------------------------
# 5. Node - Classify / route question
# --------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def classify_question(state: AgentState) -> AgentState:
    question = state["question"]
    logger.info("Classifying question")

    try:
        output = get_chains().question_router.invoke({"question": question})
        datasource = output.datasource.lower()
        if datasource in {"web_search", "websearch"}:
            datasource = "websearch"
        else:
            datasource = "vectorstore"
    except Exception:
        logger.exception("Question classification failed; defaulting to vectorstore")
        datasource = "vectorstore"

    logger.info("Category selected: %s", datasource)
    return {"datasource": datasource}


# --------------------------------------------------
# 6. Node - Retrieve context
# --------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def retrieve_context(state: AgentState) -> AgentState:
    question = state["question"]
    logger.info("Retrieving context for: %s", question)

    documents = retrieve_documents(question)
    return {"documents": documents, "question": question}


# --------------------------------------------------
# 7. Node - Grade retrieved documents
# --------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def grade_documents(state: AgentState) -> AgentState:
    question = state["question"]
    documents = list(state.get("documents") or [])
    filtered_docs: List[Document] = []

    logger.info("Grading %s retrieved documents", len(documents))

    for doc in documents:
        output = get_chains().retrieval_grader.invoke(
            {"question": question, "document": doc.page_content}
        )
        if output.score.lower() == "yes":
            filtered_docs.append(doc)

    web_search: Literal["Yes", "No"] = "Yes" if not filtered_docs else "No"
    logger.info("Relevant documents: %s; web_search=%s", len(filtered_docs), web_search)
    return {"documents": filtered_docs, "question": question, "web_search": web_search}


# --------------------------------------------------
# 8. Node - Web search
# --------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def web_search(state: AgentState) -> AgentState:
    question = state["question"]
    documents = list(state.get("documents") or [])
    logger.info("Running web search for: %s", question)

    docs = get_chains().web_search_tool.invoke({"query": question})

    if docs and isinstance(docs, list):
        if isinstance(docs[0], dict):
            web_results = "\n".join(d.get("content", str(d)) for d in docs)
        else:
            web_results = "\n".join(str(d) for d in docs)
    else:
        web_results = str(docs)

    return {
        "documents": documents + [Document(page_content=web_results)],
        "question": question,
    }


# --------------------------------------------------
# 9. Node - Generate answer
# --------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def generate_answer(state: AgentState) -> AgentState:
    question = state["question"]
    documents = list(state.get("documents") or [])
    retries = state.get("retries", 0)
    context = "\n\n".join(doc.page_content for doc in documents)

    logger.info("Generating answer (attempt %s)", retries + 1)

    generation = get_chains().rag_chain.invoke({"context": context, "question": question})
    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "retries": retries + 1,
    }


# --------------------------------------------------
# 10. Node - Rewrite query (loop)
# --------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def rewrite_query(state: AgentState) -> AgentState:
    question = state["question"]
    documents = list(state.get("documents") or [])
    logger.info("Rewriting question for retry")

    better_question = get_chains().question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}


# --------------------------------------------------
# 11. Routing logic (conditional edges, not nodes)
# --------------------------------------------------

def route_question(state: AgentState) -> Literal["retrieve", "websearch"]:
    if state.get("datasource") == "websearch":
        return "websearch"
    return "retrieve"


def decide_to_generate(state: AgentState) -> GenerateOrSearch:
    if state.get("web_search") == "Yes":
        return "websearch"
    return "generate"


def grade_generation(state: AgentState) -> GradeDecision:
    retries = state.get("retries", 0)
    if retries >= MAX_RETRIES:
        logger.warning("Max retries reached; completing workflow")
        return "max_retries"

    documents = list(state.get("documents") or [])
    generation = state["generation"]
    documents_text = "\n\n".join(doc.page_content for doc in documents)
    chains = get_chains()

    hallucination_output = chains.hallucination_grader.invoke(
        {"documents": documents_text, "generation": generation}
    )
    if hallucination_output.score.lower() != "yes":
        logger.info("Answer not grounded; looping")
        return "not supported"

    answer_output = chains.answer_grader.invoke(
        {"question": state["question"], "generation": generation}
    )
    if answer_output.score.lower() == "yes":
        logger.info("Answer accepted")
        return "useful"

    logger.info("Answer not useful; looping")
    return "not useful"


# --------------------------------------------------
# 12. Build LangGraph
# --------------------------------------------------

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("classify", classify_question)
    graph.add_node("retrieve", retrieve_context)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("websearch", web_search)
    graph.add_node("generate", generate_answer)
    graph.add_node("rewrite_query", rewrite_query)

    graph.add_edge(START, "classify")

    graph.add_conditional_edges(
        "classify",
        route_question,
        {
            "retrieve": "retrieve",
            "websearch": "websearch",
        },
    )

    graph.add_edge("retrieve", "grade_documents")

    graph.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate": "generate",
            "websearch": "websearch",
        },
    )

    graph.add_edge("websearch", "generate")

    graph.add_conditional_edges(
        "generate",
        grade_generation,
        {
            "useful": END,
            "max_retries": END,
            "not useful": "rewrite_query",
            "not supported": "rewrite_query",
        },
    )

    graph.add_edge("rewrite_query", "classify")

    return graph.compile()


@lru_cache(maxsize=1)
def get_app_graph():
    logger.info("Compiling LangGraph workflow")
    return build_graph()


def ask(question: str) -> AgentState:
    return get_app_graph().invoke({"question": question, "retries": 0})


async def aask(question: str) -> AgentState:
    return await get_app_graph().ainvoke({"question": question, "retries": 0})


class KnowledgeTransferAgent:
    """Compatibility wrapper around the compiled graph."""

    def __init__(self) -> None:
        self.graph = get_app_graph()

    def run(self, question: str):
        return self.graph.stream({"question": question, "retries": 0})
