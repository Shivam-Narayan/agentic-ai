from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import get_llm, get_retriever, get_web_search_tool
from .prompts import (
    ANSWER_GRADER_PROMPT,
    GENERATOR_PROMPT,
    HALLUCINATION_GRADER_PROMPT,
    RETRIEVAL_GRADER_PROMPT,
    ROUTER_PROMPT,
    REWRITE_PROMPT,
    GraderOutput,
    RouterOutput,
)

MAX_RETRIES = 3


# ==========================================
# 1. GRAPH STATE
# ==========================================
# The State object represents the memory of the workflow.
# As nodes execute, they can read from and append to this state.
class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: List[str]
    retries: int


class KnowledgeTransferAgent:
    def __init__(self) -> None:
        self.llm = get_llm()
        self.retriever = get_retriever()
        self.web_search_tool = get_web_search_tool()
        
        # Use structured outputs
        structured_llm_grader = self.llm.with_structured_output(GraderOutput)
        structured_llm_router = self.llm.with_structured_output(RouterOutput)

        self.retrieval_grader = RETRIEVAL_GRADER_PROMPT | structured_llm_grader
        self.rag_chain = GENERATOR_PROMPT | self.llm | StrOutputParser()
        self.hallucination_grader = HALLUCINATION_GRADER_PROMPT | structured_llm_grader
        self.answer_grader = ANSWER_GRADER_PROMPT | structured_llm_grader
        self.question_router = ROUTER_PROMPT | structured_llm_router
        self.question_rewriter = REWRITE_PROMPT | self.llm | StrOutputParser()
        
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        # ==========================================
        # 2. BUILD GRAPH
        # ==========================================
        # Initialize the StateGraph with our custom GraphState schema
        workflow = StateGraph(GraphState)

        # ==========================================
        # 3. DEFINE NODES
        # ==========================================
        # Nodes are the actual Python functions that execute steps.
        workflow.add_node("websearch", self.web_search)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("generate", self.generate)
        workflow.add_node("rewrite_query", self.rewrite_query)

        # ==========================================
        # 4. DEFINE EDGES & ROUTING
        # ==========================================
        # The entry point dynamically routes the first question
        workflow.set_conditional_entry_point(
            self.route_question,
            {
                "websearch": "websearch",
                "vectorstore": "retrieve",
            },
        )

        # Unconditional edge: After retrieval, always grade the documents
        workflow.add_edge("retrieve", "grade_documents")
        
        # Conditional edge: After grading, decide whether to generate an answer or fall back to web search
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "websearch": "websearch",
                "generate": "generate",
            },
        )
        
        # Unconditional edge: After web search, always try generating an answer
        workflow.add_edge("websearch", "generate")
        
        # Conditional edge: After generation, self-reflect on the answer's quality
        workflow.add_conditional_edges(
            "generate",
            self.grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "rewrite_query",
                "max_retries": END,
            },
        )
        
        # Unconditional edge: After rewriting a bad query, proceed to web search
        workflow.add_edge("rewrite_query", "websearch")

        # Compile the graph into an executable LangChain Runnable
        return workflow.compile()

    # ==========================================
    # 5. NODE IMPLEMENTATIONS
    # ==========================================
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def retrieve(self, state):
        question = state["question"]
        nodes = self.retriever.retrieve(question)
        documents = [Document(page_content=node.node.text) for node in nodes]
        return {"documents": documents, "question": question}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate(self, state):
        question = state["question"]
        documents = state["documents"]
        docs_content = "\n\n".join(doc.page_content for doc in documents)
        generation = self.rag_chain.invoke({"context": docs_content, "question": question})
        return {"documents": documents, "question": question, "generation": generation}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def grade_documents(self, state):
        question = state["question"]
        documents = state["documents"]
        filtered_docs = []
        web_search = "No"

        for doc in documents:
            output = self.retrieval_grader.invoke({"question": question, "document": doc.page_content})
            if output.score.lower() == "yes":
                filtered_docs.append(doc)
            else:
                web_search = "Yes"

        return {"documents": filtered_docs, "question": question, "web_search": web_search}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def web_search(self, state):
        question = state["question"]
        documents = state.get("documents", [])
        docs = self.web_search_tool.invoke({"query": question})

        if docs and isinstance(docs, list):
            if isinstance(docs[0], dict):
                web_results = "\n".join([d.get("content", str(d)) for d in docs])
            else:
                web_results = "\n".join([str(d) for d in docs])
        else:
            web_results = str(docs)

        web_results_doc = Document(page_content=web_results)
        documents.append(web_results_doc)
        return {"documents": documents, "question": question}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def route_question(self, state):
        question = state["question"]
        try:
            output = self.question_router.invoke({"question": question})
            datasource = output.datasource.lower()
            if datasource == "web_search":
                return "websearch"
            return "vectorstore"
        except Exception:
            return "vectorstore"

    def decide_to_generate(self, state):
        web_search = state.get("web_search", "No")
        if web_search == "Yes":
            return "websearch"
        return "generate"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def grade_generation_v_documents_and_question(self, state):
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        documents_text = "\n\n".join(doc.page_content for doc in documents)

        retries = state.get("retries", 0)
        
        if retries >= MAX_RETRIES:
            # We have looped too many times, terminate gracefully
            return "max_retries"

        hallucination_output = self.hallucination_grader.invoke({"documents": documents_text, "generation": generation})
        if hallucination_output.score.lower() != "yes":
            return "not supported"

        answer_output = self.answer_grader.invoke({"question": question, "generation": generation})
        if answer_output.score.lower() == "yes":
            return "useful"
        return "not useful"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def rewrite_query(self, state):
        question = state["question"]
        documents = state["documents"]
        retries = state.get("retries", 0)
        
        better_question = self.question_rewriter.invoke({"question": question})
        return {"documents": documents, "question": better_question, "retries": retries + 1}

    def run(self, question: str):
        return self.graph.stream({"question": question, "retries": 0})
