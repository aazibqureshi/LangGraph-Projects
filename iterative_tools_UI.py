"""
LinkedIn Post Generator - Colourful Streamlit UI
LangGraph iterative workflow: Writer (+Tavily search) -> Reviewer -> loop until approved

"""
import os
from typing import TypedDict, Annotated

import streamlit as st
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

load_dotenv()

# ─────────────────────────────────────────────────────────────
# 1. LANGGRAPH BACKEND
# ─────────────────────────────────────────────────────────────
search_tool = TavilySearch(max_results=3)
tools = [search_tool]

writer_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)
writer_llm_with_tools = writer_llm.bind_tools(tools)      # writer can call Tavily
reviewer_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2)


class State(TypedDict):
    topic: str
    messages: Annotated[list, add_messages]
    draft: str
    review_feedback: str
    is_approved: bool
    attempt: int
    max_attempts: int          # now configurable from the UI slider


WRITER_SYSTEM_PROMPT = (
    "You are an expert LinkedIn content writer. Write engaging, professional "
    "LinkedIn posts about the given topic. If the topic needs up-to-date "
    "information, statistics or trends, use the web search tool first. "
    "If you received feedback on a previous draft, address every point. "
    "Rules: strong hook in the first line, 1 clear takeaway, short paragraphs, "
    "around 150-200 words, end with a question or call-to-action. No hashtags."
)

REVIEWER_SYSTEM_PROMPT = (
    "You are a strict LinkedIn content reviewer. Judge whether a post is "
    "publish-ready against these criteria:\n"
    "1. Strong hook in the first line\n2. One clear, valuable takeaway\n"
    "3. Easy to skim - short paragraphs\n4. Roughly 150-200 words\n"
    "5. Ends with an engaging question or CTA\n"
    "6. Professional but human tone\n7. No hashtags\n\n"
    "Respond in exactly this format:\n"
    "VERDICT: APPROVED or REJECTED\n"
    "FEEDBACK: <one short paragraph explaining why>\n\n"
    "Approve only if ALL criteria are met."
)


def writer_node(state: State) -> dict:
    """Writes (or rewrites) the post. May call Tavily; graph then returns here."""
    msgs = state.get("messages", [])

    # FIX: if the last message is a tool result, we are CONTINUING the same
    # attempt - feed the search results back to the LLM (original code never did).
    if msgs and getattr(msgs[-1], "type", "") == "tool":
        # keep only messages of the current attempt (from last human msg onward)
        start = max(i for i, m in enumerate(msgs) if m.type == "human")
        response = writer_llm_with_tools.invoke(
            [("system", WRITER_SYSTEM_PROMPT)] + msgs[start:]
        )
        return {"messages": [response]}

    # Otherwise: brand-new attempt
    attempt = state.get("attempt", 0) + 1
    topic = state["topic"]
    if attempt == 1:
        user_message = (f"Write a LinkedIn post on this topic: {topic}. "
                        "If you need current info, search the web first.")
    else:
        user_message = (f"Your previous draft on '{topic}' was rejected. "
                        f"Reviewer feedback:\n\n{state['review_feedback']}\n\n"
                        "Write a new, improved draft that fixes every issue.")
    response = writer_llm_with_tools.invoke(
        [("system", WRITER_SYSTEM_PROMPT), ("human", user_message)]
    )
    return {"messages": [("human", user_message), response], "attempt": attempt}


tool_node = ToolNode(tools)   # executes Tavily calls requested by the writer


def extract_draft_node(state: State) -> dict:
    """Last AI message (no tool calls) = the final draft text."""
    return {"draft": state["messages"][-1].content}


def reviewer_node(state: State) -> dict:
    """LLM reviewer: returns approval flag + feedback."""
    response = reviewer_llm.invoke([
        ("system", REVIEWER_SYSTEM_PROMPT),
        ("human", f"Review this LinkedIn post draft:\n{state['draft']}\nGive your review."),
    ])
    text = response.content.strip()
    is_approved = "APPROVED" in text.upper().split("FEEDBACK")[0] \
        and "REJECTED" not in text.upper().split("FEEDBACK")[0]
    feedback = text.split("FEEDBACK:", 1)[1].strip() if "FEEDBACK:" in text else text
    return {"review_feedback": feedback, "is_approved": is_approved}


# ── routers ──
def should_use_tool(state: State):
    """Writer asked for a search? -> tools, else extract the draft."""
    return "tools" if getattr(state["messages"][-1], "tool_calls", None) else "extract_draft"


def should_loop(state: State):
    """Approved or out of attempts -> END, else rewrite."""
    if state["is_approved"] or state["attempt"] >= state["max_attempts"]:
        return END
    return "writer"


@st.cache_resource   # compile the graph only once per session
def build_app():
    g = StateGraph(State)
    g.add_node("writer", writer_node)
    g.add_node("tools", tool_node)
    g.add_node("extract_draft", extract_draft_node)
    g.add_node("reviewer", reviewer_node)
    g.add_edge(START, "writer")
    g.add_conditional_edges("writer", should_use_tool)
    g.add_edge("tools", "writer")            # FIX: search results go BACK to writer
    g.add_edge("extract_draft", "reviewer")
    g.add_conditional_edges("reviewer", should_loop)
    return g.compile()


# ─────────────────────────────────────────────────────────────
# 2. UI STYLING (CSS: animated gradient, glass cards, hover buttons)
# ─────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

/* Animated moving gradient background */
.stApp {
  background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #1d2b64, #4a1c6d);
  background-size: 400% 400%;
  animation: bgShift 18s ease infinite;
}
@keyframes bgShift { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }

/* Hero title with shimmering gradient text */
.hero { text-align:center; padding: 1.5rem 0 .5rem; animation: fadeDown .9s ease; }
.hero h1 {
  font-size: 3rem; font-weight: 800; margin: 0;
  background: linear-gradient(90deg,#ff6ec4,#7873f5,#4ade80,#facc15,#ff6ec4);
  background-size: 300% auto; -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  animation: shine 6s linear infinite;
}
.hero p { color:#cbd5e1; font-size:1.05rem; }
@keyframes shine { to { background-position: 300% center; } }
@keyframes fadeDown { from{opacity:0; transform:translateY(-25px)} to{opacity:1; transform:none} }
@keyframes fadeUp   { from{opacity:0; transform:translateY(25px)}  to{opacity:1; transform:none} }

/* Glassmorphism cards */
.card {
  background: rgba(255,255,255,.07); backdrop-filter: blur(14px);
  border: 1px solid rgba(255,255,255,.15); border-radius: 18px;
  padding: 1.2rem 1.4rem; margin: .8rem 0; color:#f1f5f9;
  animation: fadeUp .6s ease; transition: transform .25s, box-shadow .25s;
}
.card:hover { transform: translateY(-4px); box-shadow: 0 12px 30px rgba(120,115,245,.35); }
.card.ok  { border-left: 6px solid #4ade80; }
.card.bad { border-left: 6px solid #f43f5e; }

/* Pipeline steps */
.pipe { display:flex; justify-content:center; gap:14px; flex-wrap:wrap; margin:1rem 0; }
.step {
  padding:.7rem 1.1rem; border-radius:14px; color:#94a3b8; font-weight:600;
  background: rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.12);
  transition: all .3s;
}
.step.done   { color:#052e16; background:#4ade80; box-shadow:0 0 14px #4ade8088; }
.step.active { color:#fff; background:linear-gradient(135deg,#ff6ec4,#7873f5);
               animation: pulse 1s ease-in-out infinite; }
@keyframes pulse { 0%,100%{transform:scale(1); box-shadow:0 0 0 0 #ff6ec488}
                   50%{transform:scale(1.08); box-shadow:0 0 22px 6px #7873f588} }

/* Buttons: gradient + lift + glow on hover, shrink on click */
div.stButton > button {
  width:100%; border:none; border-radius:14px; padding:.8rem 1rem;
  font-weight:700; font-size:1.05rem; color:white;
  background: linear-gradient(135deg,#ff6ec4,#7873f5,#22d3ee); background-size:200% 200%;
  transition: transform .2s ease, box-shadow .2s ease, background-position .5s ease;
}
div.stButton > button:hover  { transform: translateY(-3px) scale(1.03);
  background-position: 100% 0; box-shadow: 0 10px 28px rgba(120,115,245,.6); color:white; }
div.stButton > button:active { transform: scale(.96); }

/* Inputs */
.stTextInput input { border-radius:12px !important; }
label, .stMarkdown, .stSlider { color:#e2e8f0 !important; }
</style>
"""

# (emoji, label, langgraph node name)
STEPS = [("✍️", "Writer", "writer"), ("🔍", "Web Search", "tools"),
         ("📝", "Draft", "extract_draft"), ("🧐", "Reviewer", "reviewer")]


def pipeline_html(done: set, active: str | None) -> str:
    """Builds the animated pipeline tracker: grey=waiting, pulsing=active, green=done."""
    out = '<div class="pipe">'
    for emoji, label, node in STEPS:
        cls = "active" if node == active else ("done" if node in done else "")
        out += f'<div class="step {cls}">{emoji} {label}</div>'
    return out + "</div>"


# ─────────────────────────────────────────────────────────────
# 3. PAGE
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="LinkedIn Post Generator", page_icon="🚀", layout="centered")
st.markdown(CSS, unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🚀 LinkedIn Post Generator</h1>'
            '<p>Write → Review → Improve. Loop until your post is publish-ready.</p></div>',
            unsafe_allow_html=True)

# Missing API keys? Warn early instead of crashing mid-run.
missing = [k for k in ("GROQ_API_KEY", "TAVILY_API_KEY") if not os.getenv(k)]
if missing:
    st.error(f"Missing in .env: {', '.join(missing)}")

topic = st.text_input("💡 Topic", placeholder="e.g. Why RAG beats fine-tuning for small teams")
max_attempts = st.slider("🔁 Max attempts", 1, 5, 3)

if st.button("✨ Generate Post", disabled=bool(missing)):
    if not topic.strip():
        st.warning("Pehle topic likhiye 🙂")
        st.stop()

    app = build_app()
    tracker = st.empty()        # placeholder redrawn as nodes finish
    log = st.container()        # attempt-by-attempt feedback cards
    done, active = set(), "writer"
    tracker.markdown(pipeline_html(done, active), unsafe_allow_html=True)

    # NOTE: fixed typo from original ("tpoic") - that caused a KeyError.
    initial = {"topic": topic.strip(), "messages": [], "draft": "", "review_feedback": "",
               "is_approved": False, "attempt": 0, "max_attempts": max_attempts}

    final = dict(initial)
    with st.spinner("Agents kaam kar rahe hain..."):
        # stream_mode="updates" -> one event per finished node
        for event in app.stream(initial, config={"recursion_limit": 40}, stream_mode="updates"):
            for node, update in event.items():
                final.update({k: v for k, v in update.items() if k != "messages"})
                done.add(node)
                active = None
                if node == "reviewer":              # attempt finished -> show verdict card
                    ok = update["is_approved"]
                    log.markdown(
                        f'<div class="card {"ok" if ok else "bad"}"><b>Attempt {final["attempt"]}: '
                        f'{"✅ APPROVED" if ok else "❌ REJECTED"}</b><br>{update["review_feedback"]}</div>',
                        unsafe_allow_html=True)
                    if not ok and final["attempt"] < max_attempts:
                        done, active = set(), "writer"   # next loop -> reset tracker
                tracker.markdown(pipeline_html(done, active), unsafe_allow_html=True)

    # ── final result ──
    tracker.markdown(pipeline_html({s[2] for s in STEPS}, None), unsafe_allow_html=True)
    if final["is_approved"]:
        st.balloons()
        st.success("Post approved! 🎉")
    else:
        st.warning("Max attempts reach ho gaye - best available draft neeche hai.")

    st.markdown("### 📄 Final Post")
    st.code(final["draft"], language=None)   # built-in copy button at top-right
    c1, c2 = st.columns(2)
    c1.metric("Attempts", final["attempt"])
    c2.metric("Words", len(final["draft"].split()))