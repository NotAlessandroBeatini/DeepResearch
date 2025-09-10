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
from langchain_community.tools import TavilySearchResults
import re
from bs4 import BeautifulSoup 

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
    step_general_count: int
    plan: Plan 
    step_idx: int
    done: bool

class ResearchOutput(BaseModel):
    """The final, clean output of the entire research process."""
    topic: str
    facts: List[Fact]

###### TOOLS ##################

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



class WikipediaSearchInput(BaseModel):
    query: str = Field(description="The search query for general knowledge on Wikipedia.")

def wikipedia_search_fn(query:str , config: RunnableConfig):

    cfg = (config or {}).get("configurable", {})
    top_k: int = int(cfg.get("wikipedia_top_k", 3))
    lang = cfg.get("wikipedia_lang", "en")
    wrapper = WikipediaAPIWrapper(top_k_results=top_k, lang=lang)
    return wrapper.run(query)

runnable_wiki_search = RunnableLambda(wikipedia_search_fn).with_retry(wait_exponential_jitter=True,stop_after_attempt=3)

def wikipedia_tool_fn(query:str , config: RunnableConfig):

    return runnable_wiki_search.invoke(query, config=config)

wikipedia_tool = StructuredTool.from_function(
    func=wikipedia_tool_fn,
    name="wikipedia_search",
    description="Search Wikipedia for general knowledge.",
    args_schema=WikipediaSearchInput,
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


@tool
def add_general_reference_tool(
    reference: Reference,
    tool_call_id: Annotated[str, InjectedToolCallId],  # let LG inject call id
) -> Command:
    """
    Adds an informal web source (blog, docs, tutorial, Wikipedia, etc.) to the research blackboard.
    Call this immediately after finding a relevant non-scholarly source.
    """
    fact = Fact(kind=FactKind.general_reference, content=reference)
    return Command(update={
        # ack so the agent sees its tool call confirmed
        "messages": [ToolMessage(
            content=f"Added general reference: {reference.title}",
            tool_call_id=tool_call_id,
            name="add_general_reference_tool",
        )],
        # buffer; your fact_gate will admit/limit
        "pending_facts": [fact],
    })

################################


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
    DO NOT ADD IN THE PLAN ANY STEP THAT PRODUCES A FINAL REPORT,SYNTHESIS,EXECUTIVE SUMMARY OR SIMILAR
    
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
                           "step_general_count": 0,
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
    # Collect enriched papers
    latest_cit: dict[str, Citation] = {}
    for f in state.get("facts", []):
        if f.kind != FactKind.scholar_reference: continue
        c: Citation = f.content  # type: ignore
        key = c.arxiv_id or (c.title or str(id(c)))
        latest_cit[key] = c

    enriched_papers = [c for c in latest_cit.values() if c.extended_summary and c.extended_summary.strip()]

    # Collect enriched web/wiki (using snippet as extended text)
    enriched_refs: list[Reference] = []
    for f in state.get("facts", []):
        if f.kind != FactKind.general_reference: continue
        r: Reference = f.content  # type: ignore
        if r.snippet and len(r.snippet.strip()) > 80:  # crude threshold
            enriched_refs.append(r)

    if not (enriched_papers or enriched_refs):
        final_md = f"### Synthesis for: {state.get('topic','')}\n\n(No enriched summaries available yet.)"
        return {"facts": [Fact(kind=FactKind.draft, content=final_md, confidence=0.3)]}

    synth_llm = ChatOpenAI(model="gpt-5-nano", temperature=0)
    synth_prompt = ChatPromptTemplate.from_template(
        ("You are a senior researcher. Using ONLY the extended summaries below, produce a structured, technical synthesis that answers the Topic.\n\n"
         "Requirements:\n"
         "- Organize with short headers (Context, Techniques, Evidence, Limitations, Implications).\n"
         "- Cite sources inline with bracketed numbers [1], [2], ... and list them at the end.\n"
         "- Prefer concrete details (architectures, training tricks, datasets, metrics) over generalities.\n"
         "- Keep it focused and grounded in the provided summaries.\n\n"
         "Topic: {topic}\n\n"
         "Extended Summaries:\n{summaries}\n\n"
         "Sources:\n{sources}\n\n"
         "Return ONLY the final markdown.")
    )

    lines, srcs = [], []
    idx = 0

    for c in enriched_papers:
        idx += 1
        title = c.title or "Untitled"
        venue = c.venue or ""
        year  = c.year or ""
        url   = str(c.url) if c.url else ""
        lines.append(f"[{idx}] {title} ({venue} {year}) — {c.extended_summary}")
        srcs.append(f"[{idx}] {title} ({venue} {year}) {url}")

    for r in enriched_refs:
        idx += 1
        title = r.title or "Untitled"
        pub   = r.publisher or ""
        year  = r.year or ""
        url   = str(r.url) if r.url else ""
        lines.append(f"[{idx}] {title} ({pub} {year}) — {r.snippet}")
        srcs.append(f"[{idx}] {title} ({pub} {year}) {url}")

    msg = synth_prompt.format_messages(
        topic=state.get("topic", ""),
        summaries="\n\n".join(lines)[:12000],
        sources="\n".join(srcs)[:4000],
    )
    final_md = synth_llm.invoke(msg).content
    return {"facts": [Fact(kind=FactKind.draft, content=str(final_md), confidence=0.8)]}

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

def _clean_html(html: str) -> str:
    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        text = soup.get_text(separator="\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text
    except Exception:
        # ultra-fallback: strip tags
        return re.sub(r"<[^>]+>", " ", html)

def _fetch_url_text(url: str, timeout: int = 15) -> tuple[str, str]:
    """
    Best-effort fetch of arbitrary URL. Returns (text, normalized_url).
    Handles PDFs (via PyPDF2) and HTML (via BeautifulSoup). Returns "" on failure.
    """
    try:
        import requests  # type: ignore
        r = requests.get(url, timeout=timeout, headers={"User-Agent":"Mozilla/5.0"})
        ct = r.headers.get("Content-Type","").lower()
        norm = url
        if "pdf" in ct or norm.lower().endswith(".pdf"):
            return _try_extract_pdf_text(r.content), norm
        if r.status_code == 200 and r.text:
            return _clean_html(r.text), norm
    except Exception:
        pass
    return "", url




def expander(state: ResearchState):
    """
    Enrich BOTH scholar_reference (papers) and general_reference (web/Wikipedia).
    For papers: fill `extended_summary`.
    For web/wiki: overwrite `snippet` with a focused ~150–300 word technical summary.
    """
    topic = state.get("topic") or state.get("plan", Plan(main_query="", steps=[], success_criteria=[])).main_query
    expand_llm = ChatOpenAI(model="gpt-5-nano", temperature=0)

    prompt = ChatPromptTemplate.from_template(
        ("You are a technical reader. Given the Topic and a single Source (paper or web/wiki), "
        "write a focused extended summary (200–500 words) tailored to the Topic.\n\n"
        "Rules:\n"
        "- Ground ONLY in the provided Context. Do NOT add facts or numbers that are not present there.\n"
        "- Name concrete methods/architectures/datasets/tasks and any reported metrics *only if shown in Context*.\n"
        "- Briefly explain why this source matters for the Topic (1–2 sentences).\n"
        "- End with up to 3 bullets under 'Key Details:' using numbers that appear in Context; "
        "if none exist, write 'Key Details: (no concrete metrics reported)'.\n"
        "- Keep a neutral, technical tone. No citations, links, or headings beyond 'Key Details:'.\n"
        "- If the Context is insufficient (<200 characters of substantive text), return exactly 'INSUFFICIENT_CONTEXT'.\n\n"
        "Inputs:\n"
        "Topic: {topic}\n"
        "Title: {title}\n"
        "Year/Venue/Publisher: {venue}\n"
        "URL: {url}\n\n"
        "Context (paper text / article text / abstract; may be noisy/partial):\n"
        "---\n{context}\n---\n"
        "Return ONLY the summary text or 'INSUFFICIENT_CONTEXT'.")
    )

    new_facts: list[Fact] = []

    for fact in state.get("facts", []):
        if fact.kind == FactKind.scholar_reference and isinstance(fact.content, Citation):
            cit: Citation = fact.content

            # 1) Build context
            context, pdf_url = "", ""
            if cit.arxiv_id:
                context, pdf_url = _fetch_arxiv_pdf_text(cit.arxiv_id)
            if not context and cit.url:
                ctx2, _ = _fetch_url_text(str(cit.url))
                context = ctx2 or context
            if not context:
                query = cit.arxiv_id or (cit.title or "")
                context = _fallback_arxiv_context(query)

            # 2) Summarize
            msg = prompt.format_messages(
                topic=topic or "",
                title=cit.title or "(untitled)",
                venue=f"{cit.year or ''} / {cit.venue or ''}",
                url=str(cit.url) if cit.url else "",
                context=(context or "")[:12000],
            )
            ext = str(expand_llm.invoke(msg).content or "").strip()

            # 3) Write back
            updated = Citation(**cit.model_dump())
            if ext:
                updated.extended_summary = ext
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

        elif fact.kind == FactKind.general_reference and isinstance(fact.content, Reference):
            ref: Reference = fact.content

            context = ""
            if ref.url:
                ctx, _ = _fetch_url_text(str(ref.url))
                context = ctx or ""

            # Fallback to existing snippet if we couldn’t fetch
            if not context:
                context = ref.snippet or ""

            # Skip if we have literally no content to expand
            if not context.strip():
                continue

            msg = prompt.format_messages(
                topic=topic or "",
                title=ref.title or "(untitled)",
                venue=f"{ref.year or ''} / {ref.publisher or ''}",
                url=str(ref.url) if ref.url else "",
                context=context[:12000],
            )
            ext = str(expand_llm.invoke(msg).content or "").strip()

            updated = Reference(**ref.model_dump())
            if ext:
                # reuse `snippet` as the "extended" summary to avoid schema changes
                updated.snippet = ext

            new_facts.append(Fact(
                kind=FactKind.general_reference,
                content=updated,
                confidence=0.7
            ))

    if not new_facts:
        return {}
    return {"facts": new_facts} 


###### DEDUP LOGIC ##############

def _fact_key(f: Fact) -> Optional[str]:
    # Overwrite only scholar_reference items; others just append.
    if f.kind == FactKind.scholar_reference and isinstance(f.content, Citation):
        return _citation_key(f.content)  # arxiv_id / doi / url / (title|year)
    if f.kind == FactKind.general_reference and isinstance(f.content, Reference):
        return _reference_key(f.content)
    return None

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

def _reference_key(r: Reference) -> str:
    if r.url:
        u = str(r.url).strip().lower()
        try:
            p = urlparse(u)
            u = f"{p.scheme}://{p.netloc}{p.path}"
        except Exception:
            pass
        return f"url:{u}"
    # fallback: title|year|publisher
    title = (r.title or "").strip().lower()
    year  = str(r.year or "")
    pub   = (r.publisher or "").strip().lower()
    return f"title:{title}|year:{year}|pub:{pub}"


def fact_gate(state: ResearchState, config: RunnableConfig = None):
    cfg = (config or {}).get("configurable", {}) if config else {}
    per_step_citations = int(cfg.get("max_citations_per_step", 5))
    per_step_general   = int(cfg.get("max_general_per_step", 3))

    # Build existing-key sets
    existing_sch = set()
    existing_gen = set()
    for f in state.get("facts", []):
        if f.kind == FactKind.scholar_reference and isinstance(f.content, Citation):
            existing_sch.add(_citation_key(f.content))
        elif f.kind == FactKind.general_reference and isinstance(f.content, Reference):
            existing_gen.add(_reference_key(f.content))

    # Remaining quotas (you already track step_citation_count; do the same for general if you want)
    remaining_sch = max(0, per_step_citations - int(state.get("step_citation_count", 0)))
    # Optional: track a separate counter, else just use per-call quota
    step_general_count = int(state.get("step_general_count", 0))
    remaining_gen = max(0, per_step_general - step_general_count)

    admitted: list[Fact] = []
    seen_batch_sch = set()
    seen_batch_gen = set()

    for f in state.get("pending_facts", []):
        if f.kind == FactKind.scholar_reference and isinstance(f.content, Citation):
            key = _citation_key(f.content)
            if key in existing_sch or key in seen_batch_sch or remaining_sch <= 0:
                continue
            admitted.append(f); seen_batch_sch.add(key); existing_sch.add(key); remaining_sch -= 1

        elif f.kind == FactKind.general_reference and isinstance(f.content, Reference):
            key = _reference_key(f.content)
            if key in existing_gen or key in seen_batch_gen or remaining_gen <= 0:
                continue
            admitted.append(f); seen_batch_gen.add(key); existing_gen.add(key); remaining_gen -= 1

    updates = {"pending_facts": []}
    if admitted:
        updates["facts"] = admitted
        updates["step_citation_count"] = int(state.get("step_citation_count", 0)) + sum(
            1 for f in admitted if f.kind == FactKind.scholar_reference
        )
        updates["step_general_count"] = step_general_count + sum(
            1 for f in admitted if f.kind == FactKind.general_reference
        )

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

    WIKIPEDIA_PROMPT = (
        "You are a Wikipedia research agent.\n\n"
        "Scope:\n"
        "- Use ONLY the `wikipedia_search` tool to fetch neutral, general-knowledge overviews, canonical definitions, dates, and short lists.\n"
        "- Prefer precise article titles, sections, and disambiguation handling. Do NOT invent URLs or facts.\n"
        "- When you find a solid page, log it with `add_general_reference_tool` (title, url, publisher='Wikipedia', year, snippet).\n"
        "- Add AT MOST 3 references via `add_general_reference_tool`.\n\n"
        "How to work:\n"
        "- Use the current PLAN STEP Task/Expected Output and Topic to shape your query.\n"
        "- If multiple plausible pages exist, list the top 2–3 with 1-line relevance notes.\n"
        "- Keep answers concise and factual; do not call other tools.\n\n"
        "Deliverable (append exactly at the end; do not judge success):\n"
        "==DELIVERABLE==\n"
        "OUTPUT_SUMMARY: <1–2 sentences>\n"
        "CHECKLIST: <mirror the Expected Output bullets>\n"
        "EVIDENCE: <1–3 short quoted snippets with article titles/sections>\n"
        "CITATIONS: <Wikipedia URLs (article or section anchors)>"
    )

    wikipedia_agent = create_react_agent(
        research_llm,
        tools=[wikipedia_tool, add_general_reference_tool],  # search + writer
        prompt=WIKIPEDIA_PROMPT,
        name="wikipedia_agent",
        pre_model_hook=None,
        state_schema=ResearchState,
    )

    # --- Web (Tavily) agent ---

    WEB_PROMPT = (
        "You are a web search agent.\n\n"
        "Scope:\n"
        "- Use ONLY the `tavily_search_results` tool to find recent or niche information (news, blogs, docs, repos, product/benchmark pages).\n"
        "- Prefer primary sources when possible. Extract key facts with titles and URLs. Do NOT speculate or fabricate.\n"
        "- When you find a strong source, log it with `add_general_reference_tool` (title, url, publisher/site, year if obvious, snippet).\n"
        "- Add AT MOST 3 references via `add_general_reference_tool`.\n\n"
        "- DO NOT MAKE UP INFORMATION. IF YOUR TOOL DOES NOT WORK, HAND OFF TO THE SUPERVISOR SAYING THAT"
        "How to work:\n"
        "- Use the current PLAN STEP Task/Expected Output and Topic to craft queries (include years/entities; try 1–2 reformulations if needed).\n"
        "- Keep it terse and actionable; do not call other tools.\n\n"
        "Deliverable (append exactly at the end; do not judge success):\n"
        "==DELIVERABLE==\n"
        "OUTPUT_SUMMARY: <1–2 sentences>\n"
        "CHECKLIST: <mirror the Expected Output bullets>\n"
        "EVIDENCE: <1–3 short quotes or data points with source names>\n"
        "CITATIONS: <list of URLs>"
    )

    tavily_tool = TavilySearchResults(max_results=5)

    web_agent = create_react_agent(
        research_llm,
        tools=[tavily_tool, add_general_reference_tool],
        prompt=WEB_PROMPT,          # ← fixed (was 'prompr')
        name="web_search_agent",
        pre_model_hook=None,
        state_schema=ResearchState,
    )


    SUPERVISOR_PROMPT = (
        "You are a supervisor. You will be given exactly one PLAN STEP at a time "
        "with a Task and an Expected Output.\n\n"
        "Agents:\n"
        "- arxiv_agent: academic papers/preprints; formal citations; use for scholarly evidence.\n"
        "- wikipedia_agent: canonical general knowledge; definitions, timelines, short lists.\n"
        "- web_search_agent: recent/niche web info (news, blogs, repos, docs, benchmarks).\n\n"
        "Rules:\n"
        "1) Delegate the step to exactly one agent that best fits the step.\n"
        "2) Do NOT do the work yourself; only instruct agents.\n"
        "3) Keep your instruction concise and specific to THIS step; reference the Topic if helpful.\n"
        "4) Require the workers AND ONLY THE WORKERS to end with the ==DELIVERABLE== trailer; you will judge success separately.\n"
        "5) If a task does not fit any of the workers, hand off to the next node "
    )


    supervisor = create_supervisor(
        model=research_llm,
        agents=[arxiv_agent, wikipedia_agent, web_agent],
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

    hard_question = """
i want to know how much the market for PLC software like siemens step 7 is worth. 
More specifically, i would like to know how much money a startup like xelerit could realistically make if it was a little successfull.
Xelerit is doing a sort of "cursor" for PLC
"""

    print("\n[TEST A] graph.stream() with per-step reset")
    init_state_a: ResearchState = {
        "messages": [],
        "facts": [],
        "pending_facts": [],
        "step_citation_count": 0,
        "step_general_count": 0,
        "topic": hard_question , #"what are the latest developments in RAG?",
        "plan": Plan(main_query="PLACEHOLDER", steps=[], success_criteria=[]),
        "step_idx": 0,
        "done": False,
    }
    prev_msg_id = None

    for node, state in graph.stream(
        init_state_a,
        config={"recursion_limit": 500,
                "configurable": {
                                "max_citations_per_step": 4,}},
        stream_mode="values",
        subgraphs=True
    ):
        print("NODE:", node)
        # last = (state.get("messages") or [])[-1] if state.get("messages") else None
        # last_id = getattr(last, "id", None) or id(last)
        # if last and last_id != prev_msg_id:
        #     last.pretty_print()
        #     prev_msg_id = last_id
        # pend = state.get("pending_facts") or []
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