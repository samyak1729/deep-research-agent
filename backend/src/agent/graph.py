import os

from agent.tools_and_schemas import SearchQueryList, Reflection
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig
from tavily import TavilyClient
from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from agent.configuration import Configuration
from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
)
from mirascope import llm, prompt_template
from agent.utils import (
    get_research_topic,
    insert_citation_markers,
)

load_dotenv()

if os.getenv("GEMINI_API_KEY") is None:
    raise ValueError("GEMINI_API_KEY is not set")

if os.getenv("TAVILY_API_KEY") is None:
    raise ValueError("TAVILY_API_KEY is not set")

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))





@llm.call(provider="google", model="gemini-2.5-flash", response_model=SearchQueryList)
@prompt_template(
    """
    You are a helpful assistant that generates search queries for web research.
    Current date: {current_date}

    Based on the research topic: "{research_topic}", generate {number_queries} search queries.
    Return the queries as a JSON list of strings.
    """
)
def generate_query_llm(current_date: str, research_topic: str, number_queries: int):
    pass

# Nodes
def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates search queries based on the User's question."""
    configurable = Configuration.from_runnable_config(config)

    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    current_date = get_current_date()
    research_topic = get_research_topic(state["messages"])
    number_queries = state["initial_search_query_count"]

    result = generate_query_llm(
        current_date=current_date,
        research_topic=research_topic,
        number_queries=number_queries,
    )
    return {"search_query": result.query}


def continue_to_web_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the web research node.

    This is used to spawn n number of web research nodes, one for each search query.
    """
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["search_query"])
    ]


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """LangGraph node that performs web research using the native Google Search API tool.

    Executes a web search using the native Google Search API tool in combination with Gemini 2.0 Flash.

    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including search API settings

    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
    """
    # Configure
    configurable = Configuration.from_runnable_config(config)
    formatted_prompt = web_searcher_instructions.format(
        current_date=get_current_date(),
        research_topic=state["search_query"],
    )

    # Uses the Tavily client
    response = tavily_client.search(query=state["search_query"], include_raw_content=True, timeout=30)

    sources_gathered = []
    modified_text = ""
    citations = []

    for result in response["results"]:
        source_entry = {
            "label": result["title"],
            "value": result["url"],
            "short_url": result["url"],
            "content": result["content"],
        }
        sources_gathered.append(source_entry)
        modified_text += result["content"] + "\n\n"

        # For citations, we'll use the URL and title
        # The start_index and end_index will be based on where the content appears in modified_text
        citations.append({
            "segments": [source_entry],
            "start_index": modified_text.find(result["content"]),
            "end_index": modified_text.find(result["content"]) + len(result["content"]),
            "url": result["url"],
        })

    modified_text = insert_citation_markers(modified_text, citations)

    return {
        "sources_gathered": sources_gathered,
        "search_query": [state["search_query"]],
        "web_research_result": [modified_text],
    }


@llm.call(provider="google", model="gemini-2.5-pro", response_model=Reflection)
@prompt_template(
    """
    You are a helpful assistant that analyzes search results and identifies knowledge gaps.
    Current date: {current_date}

    Based on the research topic: "{research_topic}" and the following summaries:
    {summaries}

    Determine if the information is sufficient to answer the research topic.
    If not, identify knowledge gaps and generate follow-up queries.
    Return a JSON object with "is_sufficient" (boolean), "knowledge_gap" (string), and "follow_up_queries" (list of strings).
    """
)
def reflect_llm(current_date: str, research_topic: str, summaries: str):
    pass

def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """LangGraph node that identifies knowledge gaps and generates potential follow-up queries."""
    configurable = Configuration.from_runnable_config(config)
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reasoning_model = state.get("reasoning_model", configurable.reflection_model)

    current_date = get_current_date()
    research_topic = get_research_topic(state["messages"])
    summaries = "\n\n---\n\n".join(state["web_research_result"])

    result = reflect_llm(
        current_date=current_date,
        research_topic=research_topic,
        summaries=summaries,
    )

    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": state["research_loop_count"],
        "number_of_ran_queries": len(state["search_query"]),
    }


def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> OverallState:
    """LangGraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.

    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_research_loops setting

    Returns:
        String literal indicating the next node to visit ("web_research" or "finalize_summary")
    """
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


@llm.call(provider="google", model="gemini-2.5-pro")
@prompt_template(
    """
    You are a helpful assistant that synthesizes gathered information into a coherent answer.
    Current date: {current_date}

    Based on the research topic: "{research_topic}" and the following summaries:
    {summaries}

    Synthesize the information into a well-structured research report with proper citations.
    """
)
def finalize_answer_llm(current_date: str, research_topic: str, summaries: str):
    pass

def finalize_answer(state: OverallState, config: RunnableConfig):
    """LangGraph node that finalizes the research summary."""
    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model") or configurable.answer_model

    current_date = get_current_date()
    research_topic = get_research_topic(state["messages"])
    summaries = "\n---\n\n".join(state["web_research_result"])

    result = finalize_answer_llm(
        current_date=current_date,
        research_topic=research_topic,
        summaries=summaries,
    )

    # Replace the short urls with the original urls and add all used urls to the sources_gathered
    unique_sources = []
    for source in state["sources_gathered"]:
        if source["short_url"] in result.content:
            result.content = result.content.replace(
                source["short_url"], source["value"]
            )
            unique_sources.append(source)

    return {
        "messages": [AIMessage(content=result.content)],
        "sources_gathered": unique_sources,
    }


# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Define the nodes we will cycle between
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

# Set the entrypoint as `generate_query`
# This means that this node is the first one called
builder.add_edge(START, "generate_query")
# Add conditional edge to continue with search queries in a parallel branch
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
# Reflect on the web research
builder.add_edge("web_research", "reflection")
# Evaluate the research
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)
# Finalize the answer
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="pro-search-agent")
