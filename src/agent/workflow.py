"""LangGraph: true agentic tool calling architecture."""

import logging
from typing import Annotated

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from .chains import get_llm
from .mcp_client import mcp_server_context
from .tools import search_company_documents, search_web

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    return "tools" if getattr(last_message, "tool_calls", None) else END


def build_graph(dynamic_tools: list):
    tool_node = ToolNode(dynamic_tools)
    
    def agent_node(state: AgentState) -> dict:
        llm = get_llm().bind_tools(dynamic_tools)
        response = llm.invoke(state["messages"])
        return {"messages": [response]}
        
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)

    # 1. User asks a question -> goes straight to the Agent
    builder.add_edge(START, "agent")
    
    # 2. Agent evaluates "Need a tool?"
    # If the LLM outputted tool calls, it routes to 'tools'.
    # If it didn't (e.g., generating final answer), it routes to END.
    builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    
    # 3. After the tools finish running, pass the tool result back to the Agent
    # so the LLM can read the data and generate the final answer to the user.
    builder.add_edge("tools", "agent")

    return builder.compile()


def parse_result(result: dict) -> dict:
    messages = result["messages"]
    answer = messages[-1].content

    tools_used = []
    datasource = "direct_llm"
    
    for m in messages:
        if m.type == "tool":
            tools_used.append(m.name)
            if m.name == "search_company_documents":
                datasource = "company_docs"
            elif m.name == "query-database" or m.name == "list-tables" or m.name == "read-query":
                datasource = "database"
            elif m.name == "search_web":
                datasource = "web_search"
                
    if len(tools_used) > 1:
        datasource = "multiple"
        
    return {
        "generation": answer,
        "datasource": datasource,
        "tools_used": tools_used,
    }


async def aask(question: str) -> dict:
    local_tools = [search_company_documents, search_web]
    
    async with mcp_server_context() as mcp_tools:
        all_tools = local_tools + mcp_tools
        graph = build_graph(all_tools)
        result = await graph.ainvoke({"messages": [HumanMessage(content=question)]})
        return parse_result(result)


def ask(question: str) -> dict:
    # FastAPI's async def ask_question calls aask, so we don't strictly need a synchronous ask here anymore
    # but we can implement it using asyncio.run if needed.
    import asyncio
    return asyncio.run(aask(question))


class KnowledgeTransferAgent:
    def __init__(self) -> None:
        pass

    async def run(self, question: str):
        local_tools = [search_company_documents, search_web]
        async with mcp_server_context() as mcp_tools:
            all_tools = local_tools + mcp_tools
            graph = build_graph(all_tools)
            async for s in graph.astream({"messages": [HumanMessage(content=question)]}):
                yield s
