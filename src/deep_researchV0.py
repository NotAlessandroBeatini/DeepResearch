#test_arxiv_agent.py

from typing import Union, Any
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
from typing import TypedDict, List, Optional, Annotated, Dict
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
from langchain_core.prompts import ChatPromptTemplate 
from langgraph.graph.message import RemoveMessage, REMOVE_ALL_MESSAGES
import io
from typing import Tuple
from urllib.parse import urlparse
from collections import OrderedDict

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

def _fact_key(f: Fact) -> Optional[str]:
    # Overwrite only scholar_reference items; others just append.
    if f.kind == FactKind.scholar_reference and isinstance(f.content, Citation):
        return _citation_key(f.content)  # arxiv_id / doi / url / (title|year)
    return None

def merge_facts(left: list[Fact], right: list[Fact]) -> list[Fact]:
    """
    Dedup/overwrite by key for scholar_reference facts (last write wins).
    All other fact kinds are appended in arrival order.
    Pure & deterministic (no randomness, no mutation).
    """
    keyed = OrderedDict()  # key -> Fact (latest wins)
    others: list[Fact] = []

    def ingest(lst: list[Fact]):
        for f in lst:
            k = _fact_key(f)
            if k is None:
                others.append(f)
            else:
                keyed[k] = f  # overwrite

    ingest(left)
    ingest(right)

    # Keep a stable, readable order: first keyed items, then unkeyed “others”
    return list(keyed.values()) + others

#Add a sentinel and reducer
PENDING_CLEAR = "__CLEAR__"  # simple JSON-safe token

def merge_pending_facts(left: Optional[list[Fact]], right:list[Fact]) -> list[Fact]:
    """
    Robust reducer for pending_facts:
    - Treats None as [] on either side.
    - Supports hard clear via PENDING_CLEAR.
    - Appends lists; ignores malformed updates.
    - Optionally dedupes and caps to prevent bloat.
    """
    # Normalize left
    left = left or []

    # Handle clears and no-ops
    if right is None:
        return left
    if right == PENDING_CLEAR:
        return []

    # Normal append, 
    return left + right


class ResearchState(AgentState): #TypedDict AgentState
    """The live blackboard that is passed between agents."""
    topic: str
    facts: Annotated[list[Fact], merge_facts]
    pending_facts: Annotated[list[Fact], merge_pending_facts]  # buffer written by tools
    step_citation_count: int                            # admitted citations in current step
    plan: Plan 
    step_idx: int
    done: bool

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
        "pending_facts": [fact],
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


def get_plan(topic: str) -> Plan:
    planner_prompt = ChatPromptTemplate.from_template(
    """You are a Research Planner. Your job is to take a broad research goal and turn it into a clear, step-by-step research plan.

    You do not execute the research yourself — instead, you design a sequence of tasks that other specialized researcher agents will carry out.

    ---

    Guidelines:
    1. **Understand the Goal**: Restate the user’s task in your own words to ensure clarity.  
    2. **Break Down the Problem**: Identify the major sub-questions or aspects that must be researched.  
    3. **Order the Steps**: Arrange tasks in a logical order (general context → deeper details → synthesis).  
    4. **Output Format**: Produce a structured plan as JSON with the following schema:
    - `main_query`: a single concise query that captures the essence of the research task.  
    - `steps`: a numbered list of atomic tasks. Each step should be:
        - **Task**: short description of what to find or do.  
        - **Expected Output**: what kind of result is expected (e.g., list of sources, factual summary, comparison table).  

    DO NOT ADD ANY SUPERFLOUS STEPS. KEEP THE PLAN AS SHORT AS POSSIBLE.
    
    Style constraints:
    - Each step MUST be self-contained: avoid pronouns/anaphora like "these papers" or "the above".
    - Refer explicitly to entities, e.g., "the papers collected in Step 1".
    
    ---

    Formatting Example (for the goal: “What is the impact of AI on climate modeling?”):
    ```json
    {{
    "main_query": "AI in climate modeling",
    "steps": [
        {{
        "Task": "Collect recent academic and industry publications on the use of AI/ML in climate science and climate modeling.",
        "Expected Output": "Curated corpus with metadata for each item: title, venue, year, link/DOI, and a one-sentence relevance note.",
        }},
        {{
        "Task": "Extract and organize methodologies reported across the corpus collected in Step 1.",
        "Expected Output": "Structured table mapping technique → representative papers → datasets/tasks → reported metrics and stated limitations.",
        }},
        {{
        "Task": "Identify open problems and controversies using the methodology synthesis from Step 2.",
        "Expected Output": "List of 5–10 concrete open questions or limitations, each with 1–2 supporting citations from the corpus.",
        }}
    ]
    }}
    ---

    Topic: {topic}

    Return ONLY the structured object."""
    ) 

    planner_llm = ChatOpenAI(model="gpt-5-nano",temperature=0)
    planner_llm = planner_llm.with_structured_output(Plan)

    planner_chain = planner_prompt | planner_llm

    plan_res = planner_chain.invoke({"topic":topic})
    return plan_res

# --- planner node: ensure plan exists (or regenerate if empty) ---
def planner(state: ResearchState):
    if not state.get("plan") or not state["plan"].steps:
        p = get_plan(state.get("topic", ""))  # your planner returns a Plan
        return {"plan": p, "step_idx": 0, "done": False}
    if "step_idx" not in state:
        return {"step_idx": 0, "done": False}
    return {}

# --- iterator node: pass the CURRENT step to the supervisor ---
def step_iterator(state: ResearchState):
    i = state["step_idx"]
    steps = state["plan"].steps
    if i >= len(steps):
        return Command(update={"done": True}, goto="advance")

    step = steps[i]
    msg = HumanMessage(content=(
        f"PLAN STEP {i+1}/{len(steps)}\n"
        f"Task: {step.task}\n"
        f"Expected Output: {step.expected_output}\n"
        f"Topic: {state['topic']}\n"
        "Delegate as needed. Workers must return the ==DELIVERABLE== evidence trailer; "
        "the supervisor, not the worker, will judge success.\n"
        "Do ONLY this step. Use tools. When finished, stop."
    ))
    #return Command(update={"messages": [msg]}, goto="supervisor")
    return Command(update={"messages": [msg],   #[RemoveMessage(id=REMOVE_ALL_MESSAGES), msg],
                           "step_citation_count": 0,
                           "pending_facts": PENDING_CLEAR},
                    goto="supervisor")

# --- after supervisor returns, advance the pointer (optionally log) ---
def advance(state: ResearchState):
    i = state["step_idx"]
    steps = state["plan"].steps

    updates = {}
    i += 1
    done = i >= len(steps)
    updates.update({"step_idx": i, "done": done})
    return Command(update=updates)

def finish(state: ResearchState):
    return {}

def synthesizer(state: ResearchState):
    """
    Build the final, structured answer using ONLY citations that have extended_summary.
    If duplicates exist, prefer the last (most recent enriched copy).
    """
    # Collapse to the latest per arxiv_id or title
    latest: dict[str, Citation] = {}
    for f in state.get("facts", []):
        if f.kind != FactKind.scholar_reference:
            continue
        cit: Citation = f.content  # type: ignore
        latest[_citation_key(cit)] = cit  # later ones overwrite earlier ones

    enriched = [c for c in latest.values() if (c.extended_summary and c.extended_summary.strip())]
    if not enriched:
        # Nothing enriched; bail gracefully
        final_md = f"### Synthesis for: {state.get('topic','')}\n\n(No enriched summaries available yet.)"
        return {"facts": [Fact(kind=FactKind.draft, content=final_md, confidence=0.3)]}

    synth_llm = ChatOpenAI(model="gpt-5-nano", temperature=0)
    synth_prompt = ChatPromptTemplate.from_template(
        """You are a senior researcher. Using ONLY the extended summaries below, produce a
structured, technical synthesis that answers the Topic.

Requirements:
- Organize with short headers (Context, Techniques, Evidence, Limitations, Implications).
- Cite sources inline with bracketed numbers [1], [2], ... and list them at the end.
- Prefer concrete details (architectures, training tricks, datasets, metrics) over generalities.
- Keep it focused and grounded in the provided summaries.

Topic: {topic}

Extended Summaries:
{summaries}

Sources:
{sources}

Return ONLY the final markdown."""
    )

    # Build summaries block and ordered sources
    lines = []
    srcs  = []
    for idx, c in enumerate(enriched, 1):
        title = c.title or "Untitled"
        venue = c.venue or ""
        year  = c.year or ""
        url   = str(c.url) if c.url else ""
        lines.append(f"[{idx}] {title} ({venue} {year}) — {c.extended_summary}")
        srcs.append(f"[{idx}] {title} ({venue} {year}) {url}")

    msg = synth_prompt.format_messages(
        topic=state.get("topic", ""),
        summaries="\n\n".join(lines)[:12000],
        sources="\n".join(srcs)[:4000],
    )
    final_md = synth_llm.invoke(msg).content

    return {"facts": [Fact(kind=FactKind.draft, content=str(final_md), confidence=0.8)],
            "messages": [AIMessage(content=final_md)]}

def _try_extract_pdf_text(pdf_bytes: bytes) -> str:
    """Best-effort PDF text extraction. Uses PyPDF2 if available, else returns ''."""
    try:
        import PyPDF2  # type: ignore
    except Exception:
        return ""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for i, p in enumerate(reader.pages):
            try:
                pages.append(p.extract_text() or "")
            except Exception:
                continue
        return "\n".join(pages)
    except Exception:
        return ""

def _fetch_arxiv_pdf_text(arxiv_id: str) -> Tuple[str, str]:
    """
    Try to download the arXiv PDF and extract text.
    Returns (text, url). On failure returns ("", pdf_url or "").
    """
    if not arxiv_id:
        return "", ""
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    try:
        import requests  # type: ignore
        r = requests.get(pdf_url, timeout=15)
        if r.status_code == 200 and r.content:
            text = _try_extract_pdf_text(r.content)
            return text or "", pdf_url
        return "", pdf_url
    except Exception:
        return "", pdf_url

def _fallback_arxiv_context(query: str) -> str:
    """Fallback to ArxivAPIWrapper.run() (abstract & metadata)."""
    try:
        wrapper = ArxivAPIWrapper(top_k_results=1)
        return wrapper.run(query)[:12000]  # keep prompt tame
    except Exception:
        return ""

def expander(state: ResearchState):
    """
    For each scholar_reference, read the paper (best-effort) and produce a focused,
    technical extended_summary that is relevant to the main query/topic.
    We DO NOT mutate existing facts in place (reducer is additive). Instead we append
    updated scholar_reference Facts with the extended_summary filled.
    """
    topic = state.get("topic") or state.get("plan", Plan(main_query="", steps=[], success_criteria=[])).main_query
    expand_llm = ChatOpenAI(model="gpt-5-nano", temperature=0)

    prompt = ChatPromptTemplate.from_template(
        """You are a technical reader. Write a focused extended summary (~150–300 words)
for the paper below, tailored to the research topic.

Requirements:
- Be specific: name key methods, design choices, datasets, tasks, and any reported metrics.
- Explain *how* this paper matters for the topic; avoid generic statements.
- If applicable, include 1–3 short bullets at the end labeled "Key Details:" with concrete numbers (e.g., accuracy, speedup).

Topic: {topic}
Title: {title}
Year/Venue: {year} / {venue}

Context (paper text or abstract; may be noisy/partial):
---
{context}
---
Return ONLY the extended summary text."""
    )

    new_facts: List[Fact] = []
    for fact in state.get("facts", []):
        if fact.kind != FactKind.scholar_reference:
            continue
        cit: Citation = fact.content  # type: ignore

        # 1) Try full paper text via arXiv PDF; fallback to abstract
        context = ""
        pdf_url = ""
        if cit.arxiv_id:
            context, pdf_url = _fetch_arxiv_pdf_text(cit.arxiv_id)
        if not context:
            # Fallback query preference: arxiv_id -> title
            query = cit.arxiv_id or (cit.title or "")
            context = _fallback_arxiv_context(query)

        # 2) Ask LLM for a focused extended summary
        msg = prompt.format_messages(
            topic=topic or "",
            title=cit.title or "(untitled)",
            year=cit.year or "",
            venue=cit.venue or "",
            context=(context or "")[:12000],
        )
        ext = expand_llm.invoke(msg).content
        ext = str(ext).strip()

        # 3) Append an updated Citation as a new scholar_reference Fact
        updated = Citation(**cit.model_dump())
        updated.extended_summary = ext if ext else cit.extended_summary
        # (Optional) add pdf url to url if missing and we have one
        if (not updated.url) and pdf_url:
            try:
                updated.url = pdf_url  # type: ignore
            except Exception:
                pass

        new_facts.append(Fact(
            kind=FactKind.scholar_reference,
            content=updated,
            confidence=0.8
        ))

    if not new_facts:
        return {}
    return {"facts": new_facts}

def _citation_key(c: Citation) -> str:
    """
    Stable dedupe key: prefer arxiv_id, else doi, else normalized URL, else (title|year).
    """
    if c.arxiv_id: return f"arxiv:{c.arxiv_id.strip().lower()}"
    if c.doi:      return f"doi:{c.doi.strip().lower()}"
    if c.url:
        u = str(c.url).strip().lower()
        try:
            p = urlparse(u)
            u = f"{p.scheme}://{p.netloc}{p.path}"
        except Exception:
            pass
        return f"url:{u}"
    title = (c.title or "").strip().lower()
    year = str(c.year or "")
    return f"title:{title}|year:{year}"

def fact_gate(state: ResearchState, config: RunnableConfig = None):
    """
    Admit at most K new unique scholar_reference citations *per step* from pending_facts.
    Dedupe against existing facts AND against other pendings in this call.
    """
    cfg = (config or {}).get("configurable", {}) if config else {}
    per_step_cap = int(cfg.get("max_citations_per_step", 5))

    # Build the existing key set from admitted citations
    existing_keys = set()
    for f in state.get("facts", []):
        if f.kind != FactKind.scholar_reference: continue
        cit: Citation = f.content  # type: ignore
        existing_keys.add(_citation_key(cit))

    # How many more we can accept this step
    remaining = max(0, per_step_cap - int(state.get("step_citation_count", 0)))

    admitted: List[Fact] = []
    seen_in_batch = set()

    for f in state.get("pending_facts", []):
        if f.kind != FactKind.scholar_reference: 
            continue
        cit: Citation = f.content  # type: ignore
        key = _citation_key(cit)
        if key in existing_keys or key in seen_in_batch:
            continue
        if remaining <= 0:
            break
        admitted.append(f)
        seen_in_batch.add(key)
        existing_keys.add(key)
        remaining -= 1

    updates = {"pending_facts": []}  # clear buffer (we could keep remainders if you prefer)

    if admitted:
        updates["facts"] = admitted
        updates["step_citation_count"] = int(state.get("step_citation_count", 0)) + len(admitted)

    return updates


def main():
        
    research_llm = ChatOpenAI(model="gpt-5-nano",temperature=0) #for react agent must have "free" llm, no structured output


    ARXIV_PROMPT = (
        "You are an academic research agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Focus ONLY on the current step/topic.\n"
        "- Use tools as needed, but add AT MOST 3 strong papers via add_scholar_reference_tool.\n"
      #  "- if you find relevant papers, you MUST add them with add_scholar_reference_tool before output the ==DELIVERABLE==.\n"
        "- AFTER you have added up to 3 papers OR you cannot find more high-signal items,\n"
        "  STOP CALLING TOOLS and produce a final message that starts with:\n"
        "  ==DELIVERABLE==\n"
        "  OUTPUT_SUMMARY: <1–2 sentences>\n"
        "  CHECKLIST: <bullets mirroring the Expected Output>\n"
        "  EVIDENCE: <short quotes/metrics>\n"
        "  CITATIONS: <urls or arxiv ids>\n"
        "Do not resume tool use after the ==DELIVERABLE== block."
    )
    #"- After you're done with your tasks, respond to the supervisor directly\n" 

    arxiv_agent = create_react_agent(
        research_llm,
        tools=[arxiv_tool, add_scholar_reference_tool],  # search + writer
        prompt=ARXIV_PROMPT,
        name="arxiv_agent",
        pre_model_hook= None, #can trim messages or only append the last?
        state_schema=ResearchState,
    )


    SUPERVISOR_PROMPT = (
    "You are a supervisor. You will be given exactly one PLAN STEP at a time "
    "with a Task and an Expected Output.\n\n"
    "Rules:\n"
    "1) Delegate the step to exactly one agent that can best complete it.\n"
    "2) Do NOT do the work yourself; only instruct agents.\n"
    "3) Keep your instruction concise and specific to THIS step.\n"
    )
        # "- a web search agent. Assign web-search-related tasks to this agent\n"
        # "- a wikipedia search agent. Assign wikipedia search-realate tasks to this agent\n"

    supervisor = create_supervisor(
        model=research_llm,
        agents=[arxiv_agent],
        prompt= SUPERVISOR_PROMPT,
        # optional: include workers' inner histories vs final messages, etc.
        add_handoff_back_messages=True,
        add_handoff_messages = True,
        output_mode= "last_message", # only add worker last message to the message history
        state_schema=ResearchState
    ).compile()


    wf = StateGraph(ResearchState)
    wf.add_node("planner", planner)
    wf.add_node("step_iterator", step_iterator)
    wf.add_node("supervisor", supervisor)   # compiled supervisor node
    wf.add_node("advance", advance)
    wf.add_node("finish", finish)
    wf.add_node("expander", expander)
    wf.add_node("synthesizer", synthesizer)
    wf.add_node("fact_gate", fact_gate)

    wf.add_edge(START, "planner")
    wf.add_edge("planner", "step_iterator")
    wf.add_edge("step_iterator", "supervisor")
    wf.add_edge("supervisor", "fact_gate")
    wf.add_edge("fact_gate", "advance")
    wf.add_conditional_edges(
                            "advance",
                            lambda s: "expander" if s.get("done") else "step_iterator",
                            {"expander": "expander", "step_iterator": "step_iterator"},
                            )
    wf.add_edge("expander", "synthesizer")
    wf.add_edge("synthesizer", "finish")
    wf.add_edge("finish", END)

    graph = wf.compile()
    os.makedirs("artifacts", exist_ok=True)

    # Main workflow graph
    png = graph.get_graph().draw_mermaid_png()  # returns PNG bytes
    with open("artifacts/deep_research_graph.png", "wb") as f:
        f.write(png)

    #####################
    # TEST
    #####################

    print("\n[TEST A] graph.stream() with per-step reset")
    init_state_a: ResearchState = {
        "messages": [HumanMessage(content="'Speculative RAG' (ICLR 2025).")],
        "facts": [],
        "pending_facts": [], 
        "step_citation_count": 0, 
        "topic": "what are the latest developements in RAG?",
        "plan": Plan(main_query="RAG SOTA", steps=[], success_criteria=[]),  # empty -> planner fills
        "step_idx": 0,
        "done": False,
    }

    for node, state in graph.stream(
        init_state_a,
        config={"recursion_limit": 30,
                "configurable": {
                                "max_citations_per_step": 4,}},
        stream_mode="values",
        subgraphs=True
    ):
        print("NODE:", node)
        if state.get("messages"):
            # Safe: we always keep the scratchpad tiny per step
            state["messages"][-1].pretty_print()
        pend = state.get("pending_facts") or []
        print("PENDING FACT SIZE: ", len(state.get("pending_facts")))
        print("FACTS SIZE:", len(state.get("facts", [])))

    print("TEST FINAL STATE FACTS:", len(state.get("facts", [])))
    from pprint import pprint
    pprint(state.get("facts", []))


if __name__ == "__main__":
    # IT SEEMS TO BE WORKING; HURRAY!
    load_dotenv()
    main()



"""
    SUPERVISOR_PROMPT = (
        "You are a plan-following supervisor. You receive ONE PLAN STEP at a time "
        "with a Task and an Expected Output.\n\n"
        "If there is NO worker reply yet for this step:\n"
        "- Delegate to exactly one agent with a concise, specific instruction.\n"
        "- Require the worker to end with this evidence trailer (NO judgment):\n"
        "  ==DELIVERABLE==\n"
        "  OUTPUT_SUMMARY: <1–2 sentences>\n"
        "  CHECKLIST: <1–3 bullets mirroring the Expected Output>\n"
        "  EVIDENCE: <quotes/snippets or data points>\n"
        "  CITATIONS: <urls or identifiers if applicable>\n\n"
        "If there IS a worker reply (the last message in the transcript):\n"
        "- Evaluate ONLY that reply against the Expected Output using the trailer as evidence.\n"
        "- If fully satisfied, reply EXACTLY: CONTROL: STEP_DONE — <short summary>\n"
        "- Otherwise, reply EXACTLY: CONTROL: RETRY — <what is missing in one sentence>\n"
        "  Then immediately delegate again with a single, focused instruction to close the gap.\n\n"
        "Rules:\n"
        "- Handle exactly one agent at a time.\n"
        "- Do NOT do the work yourself; only instruct and evaluate.\n"
        "- Keep responses terse."
    )

"""