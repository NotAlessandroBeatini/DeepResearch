#test_arxiv_agent.py

from typing import Union
from typing import Optional, List, TypedDict
from pydantic import BaseModel, HttpUrl, Field
from enum import Enum
import operator
from langgraph.prebuilt.chat_agent_executor import AgentState
from typing import TypedDict, List, Optional
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_community.tools import TavilySearchResults
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
import json
import uuid
import os
from dotenv import load_dotenv
from typing import TypedDict, List, Optional, Annotated
from langchain_core.messages import BaseMessage,HumanMessage, AIMessage
from langgraph.graph import START, END, StateGraph, add_messages
from pydantic import BaseModel, Field
from uuid import uuid4
from datetime import datetime
from langgraph.types import Command
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId




class FactKind(str, Enum):
    finding = "finding"
    draft = "draft"
    critique = "critique"
    plan_step = "plan_step"
    scholar_reference = "scholar_reference"        # formal academic item
    general_reference = "general_reference"      # blog/vendor docs/etc.
    note = "note"

        
class Citation(BaseModel):
    # --- Fields for the COLLECTOR agent (easy to find) ---
    title: Optional[str] = Field(None, description="Publication title.")
    url: Optional[HttpUrl] = Field(None, description="Canonical/stable link.")
    authors: Optional[List[str]] = Field(None, description="Author list.")
    venue: Optional[str] = Field(None, description="Journal/conference or 'arXiv'.")
    year: Optional[int] = Field(None, description="Publication year.")
    doi: Optional[str] = Field(None, description="DOI if available.")
    arxiv_id: Optional[str] = Field(None, description="arXiv ID like 2405.06211.")

    # --- Fields for the ENRICHMENT agent (require analysis) ---
    snippet: Optional[str] = Field(
        None, description="A brief, quoted sentence or two from the source."
    )
    contributions: Optional[List[str]] = Field(
        None, description="[ENRICHMENT] 2–5 bullets of concrete contributions/novelty."
    )
    results: Optional[List[str]] = Field(
        None, description="[ENRICHMENT] 1–4 concrete metrics or original claims."
    )
    extended_summary: Optional[str] = Field(
        None,
        description="[ENRICHMENT] ~50–300 words giving an accurate overview."
    )
    tags: Optional[List[str]] = Field(None, description="Short tags.")


class Reference(BaseModel):
    # --- Fields for the COLLECTOR agent ---
    title: str = Field(..., description="Title of blog/doc/tutorial/case study.")
    url: HttpUrl = Field(..., description="Link to the source.")
    publisher: Optional[str] = Field(None, description="Blog/company site.")
    author: Optional[str] = Field(None, description="Main author if available.")
    year: Optional[int] = Field(None, description="Year.")

    # --- Field for the ENRICHMENT agent ---
    snippet: Optional[str] = Field(
        None, 
        description="[ENRICHMENT] 50–300 words accurately summarizing the source's findings."
    )


class Finding(BaseModel):
    statement: str = Field(description="1–2 sentence synthesized insight.")                              # 1–2 sentences
    supporting_sources: Optional[List[HttpUrl]] = Field(
        None, description="sources backing this finding."
    )


class Fact(BaseModel):
    id: Optional[str] = Field(None, description="Server-generated unique identifier.")
    kind: FactKind
    content: Union[Citation, Reference, Finding, str] # Use a Union for content
    created_at: Optional[datetime] = Field(
        None, description="Server timestamp."
    )
    confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Self-estimated confidence of relevance 0–1."
    )


class Step(BaseModel):
    """A single, atomic step in a research plan."""
    task: str = Field(..., description="The specific, imperative task for the researcher to perform.")
    expected_output: str = Field(..., description="A description of the expected output from this step.")

class Plan(BaseModel):
    """A structured research plan."""
    main_query: str = Field(..., description="A single crisp query to kick off search, summarizing the topic.")
    steps: List[Step] = Field(..., description="The list of atomic, ordered steps to execute.")
    success_criteria: List[str] = Field(..., description="A checklist of criteria to verify the research is complete.")


class ResearchState(AgentState): #TypedDict AgentState
    """The live blackboard that is passed between agents."""
    topic: str
    facts: Annotated[list[Fact], operator.add]
    plan: Plan 

class ResearchOutput(BaseModel):
    """The final, clean output of the entire research process."""
    topic: str
    facts: List[Fact]


class ArxivSearchInput(BaseModel):
    query: str = Field(description="The search query for scientific papers on ArXiv.")

def arxiv_search_fn(query: str, config: RunnableConfig) -> str:
    cfg = (config or {}).get("configurable", {})
    top_k = int(cfg.get("arxiv_top_k", 3))
    wrapper = ArxivAPIWrapper(top_k_results=top_k)
    return wrapper.run(query)

arxiv_tool = StructuredTool.from_function(
    name="arxiv_search",
    description="Search ArXiv for scientific papers.",
    args_schema=ArxivSearchInput,
    func=arxiv_search_fn,
    handle_tool_error=True
)



@tool
def add_scholar_reference_tool(
    citation: Citation,
    tool_call_id: Annotated[str, InjectedToolCallId],  # REQUIRED so we can ack
) -> Command:
    """
    Adds a formal academic citation (paper, preprint, etc.) to the research blackboard.
    """
    fact = Fact(kind=FactKind.scholar_reference, content=citation)

    # Return BOTH a ToolMessage (ack) and the state update to `facts`
    return Command(update={
        "messages": [ToolMessage(
            content=f"Added scholar citation: {citation.title or '(untitled)'}",
            tool_call_id=tool_call_id,
            name="add_scholar_reference_tool",
        )],
        "facts": [fact],
    })

def _print_facts(state: ResearchState, label: str):
    print(f"\n=== {label} ===")
    facts = state.get("facts", [])
    print("FACTS COUNT:", len(facts))
    for i, f in enumerate(facts, 1):
        if isinstance(f.content, Citation):
            print(f"{i}. {f.kind}  |  {f.content.title}  |  {f.content.url}")
        else:
            print(f"{i}. {f.kind}  |  {f.content}")

# @tool
# def add_scholar_reference_tool(
#     citation: Citation,
# #    state: Annotated[ResearchState, InjectedState] = None,  # injected by LangGraph, only need it if i want to inspect state
#  #   tool_call_id: Annotated[str, InjectedToolCallId] = None,      # optional: for threading            
#     ) -> Command:
#     """
#     Adds a formal academic citation (paper, preprint, etc.) to the research blackboard.
#     Call this immediately after finding a relevant scholarly article.
#     """
#     fact = Fact(kind=FactKind.scholar_reference, content=citation)

#     # Update the graph state; reducer will append to `blackboard`
#     return Command(update={"facts": [fact]})



def main():
        
    research_llm = ChatOpenAI(model="gpt-5-nano",temperature=0) #for react agent must have "free" llm, no structured output


    ARXIV_PROMPT = (
            "You are an academic research agent.\n\n"
            "INSTRUCTIONS:\n"
            "- Assist ONLY with research-related tasks on the topic given to you.\n"
            "- For each solid paper, CALL add_scholar_reference_tool with a proper Citation.\n"
            
    )
    #"- After you're done with your tasks, respond to the supervisor directly\n" 

    arxiv_agent = create_react_agent(
        research_llm,
        tools=[arxiv_tool, add_scholar_reference_tool],  # search + writer
        prompt=ARXIV_PROMPT,
        name="arxiv_agent",
        state_schema=ResearchState
    
    )

    from langgraph.graph import StateGraph, START, END

    graph = StateGraph(ResearchState)
    graph.add_node("arxiv_agent", arxiv_agent)
    graph.add_edge(START, "arxiv_agent")
    graph.add_edge("arxiv_agent", END)
    app = graph.compile()


    # ----------------------------
    # TEST A: simple test of adding facts to the state of the graph.
    # ----------------------------
    print("\n[TEST A] invoke() with seeded messages + facts")
    init_state_a: ResearchState = {
        "messages": [HumanMessage(content="'Speculative RAG' (ICLR 2025).")],
        "facts": [],  # seeded (not strictly needed with our tolerant reducer, but good hygiene)
        # the rest of your custom fields are unused in this simple test
        "topic": "RAG SOTA",
        "plan": Plan(main_query="RAG SOTA", steps=[], success_criteria=[]),
    }

    for node, state in app.stream(
        init_state_a,
        config={"recursion_limit": 20},
        stream_mode="values",
        subgraphs=True
    ):
        print("STATE: \n", state.keys())
        print("NODE: \n", node, " \n")

        print("FACTS SIZE:", len(state.get("facts", [])))

        state["messages"][-1].pretty_print()


    # print("TEST A FINAL STATE: \n", event["facts"].content)
    print("TEST FINAL STATE FACTS: \n", len(state["facts"]) )
    print("FINAL STATE FACTS: \n", state["facts"])



if __name__ == "__main__":
    """
    IT SEEMS TO BE WORKING; HURRAY!
    """
    load_dotenv()
    main()