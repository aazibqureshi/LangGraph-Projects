from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain_core.messages import ToolMessage
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command

load_dotenv()

# ---------------------------------------------------------------
# Tools
# ---------------------------------------------------------------

search_tool = TavilySearch(max_results=3)
tools = [search_tool]

# ---------------------------------------------------------------
# LLM
# ---------------------------------------------------------------

writer_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)
writer_llm_with_tools = writer_llm.bind_tools(tools)

# ---------------------------------------------------------------
# State
# ---------------------------------------------------------------


class State(TypedDict):
    topic: str
    messages: Annotated[list, add_messages]
    draft: str
    review_feedback: str
    is_approved: bool
    attempt: int


# ---------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------

WRITER_SYSTEM_PROMPT = (
    "You are an expert LinkedIn content writer. Your job is to write "
    "engaging, professional LinkedIn posts about the given topic. "
    "If the topic requires up-to-date information, statistics, or "
    "current trends, use the web search tool to gather fresh context "
    "before writing. If you have already received feedback on a "
    "previous draft, carefully address every point in the new draft. "
    "Rules for good LinkedIn posts: strong hook in the first line, "
    "1 clear takeaway, easy to skim (short paragraphs), around "
    "150-200 words, ends with a question or call-to-action to invite "
    "engagement. Do not use hashtags."
)


def writer_node(state: State) -> dict:
    """Writes or rewrites the LinkedIn post. Can call Tavily to search first."""
    messages_so_far = state["messages"]

    # Agar last message tool ka result hai, to sirf final post likhwao
    # (attempt mat badhao, naya human message mat jodo)
    if messages_so_far and isinstance(messages_so_far[-1], ToolMessage):
        response = writer_llm_with_tools.invoke(
            [("system", WRITER_SYSTEM_PROMPT)] + messages_so_far
        )
        return {"messages": [response]}

    attempt = state.get("attempt", 0) + 1
    topic = state["topic"]
    previous_feedback = state.get("review_feedback", "")

    if attempt == 1:
        user_message = (
            f"Write a LinkedIn post on this topic: {topic}. "
            f"If you need current info, search the web first."
        )
    else:
        user_message = (
            f"Your previous draft on '{topic}' was rejected. "
            f"Here is the reviewer's feedback:\n\n{previous_feedback}\n\n"
            f"Write a new, improved draft that fixes every issue mentioned. "
            f"Do not repeat the mistake."
        )

    messages = (
        [("system", WRITER_SYSTEM_PROMPT)]
        + messages_so_far
        + [("human", user_message)]
    )
    response = writer_llm_with_tools.invoke(messages)

    return {"messages": [("human", user_message), response], "attempt": attempt}


tool_node = ToolNode(tools)


def extract_draft_node(state: State) -> dict:
    """Writer ke tool calls khatam hone ke baad final text ko draft bana do."""
    last_message = state["messages"][-1]
    draft = last_message.content
    print(f"\n\nGenerated post:\n{draft}\n")
    return {"draft": draft}


def human_review_node(state: State) -> dict:
    """Graph ko pause karta hai aur human se approve/feedback maangta hai."""
    human_response = interrupt(
        {
            "draft": state["draft"],
            "attempt": state["attempt"],
            "instruction": (
                "Type 'approved' to accept, or type your feedback "
                "to request changes."
            ),
        }
    )

    response = human_response.strip()

    if response.lower() in ["approved", "approve", "yes", "ok", "good"]:
        return {
            "is_approved": True,
            "review_feedback": "Approved by human.",
        }
    else:
        return {
            "is_approved": False,
            "review_feedback": response,
        }


# ---------------------------------------------------------------
# Router functions
# ---------------------------------------------------------------


def should_use_tool(state: State):
    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "extract_draft"


def should_use_looping(state: State):
    if state["is_approved"]:
        print("Post has been approved\n")
        return END
    if state["attempt"] >= 3:
        print("Reached max attempts")
        return END
    return "writer"


# ---------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------

graph = StateGraph(State)

graph.add_node("writer", writer_node)
graph.add_node("tools", tool_node)
graph.add_node("extract_draft", extract_draft_node)
graph.add_node("human_review", human_review_node)

graph.add_edge(START, "writer")
graph.add_conditional_edges("writer", should_use_tool)

graph.add_edge("tools", "writer")
graph.add_edge("extract_draft", "human_review")

graph.add_conditional_edges("human_review", should_use_looping)

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# ---------------------------------------------------------------
# Run
# ---------------------------------------------------------------

print("=" * 55)
print("Welcome to the LinkedIn Post Generator")
print("=" * 55)
print("\nThis tool will draft a LinkedIn post for you, and you can")
print("review it and give feedback until it's publish-ready.")
print("=" * 55)

topic = input("\nWhat topic do you want a LinkedIn post about?\n").strip()

if not topic:
    print("\nNo topic given. Exiting.")
else:
    print("\nStarting generation...\n")

    config = {"configurable": {"thread_id": "linkedin_session_1"}}

    initial_state = {
        "topic": topic,
        "messages": [],
        "draft": "",
        "review_feedback": "",
        "is_approved": False,
        "attempt": 0,
    }

    result = app.invoke(initial_state, config=config)

    while "__interrupt__" in result:
        interrupt_data = result["__interrupt__"][0].value

        print("\n" + "=" * 55)
        print(f"DRAFT FOR YOUR REVIEW (Attempt {interrupt_data['attempt']})")
        print("=" * 55)
        print(interrupt_data["draft"])
        print("=" * 55)
        print(f"\n{interrupt_data['instruction']}")

        human_input = input("\nYour response: ").strip()

        result = app.invoke(Command(resume=human_input), config=config)

    print("\n" + "=" * 55)
    print("FINAL LINKEDIN POST")
    print("=" * 55)
    print(result["draft"])
    print("=" * 55)
    print(f"Total attempts: {result['attempt']}")
    print(f"Approved by human: {result['is_approved']}")