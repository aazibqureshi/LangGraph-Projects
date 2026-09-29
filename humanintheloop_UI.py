"""
LinkedIn Post Generator - Human-in-the-Loop Streamlit UI
Flow: Writer (+Tavily) -> Draft -> PAUSE for human -> Approve / Feedback -> loop

Run:  pip install streamlit langgraph langchain-groq langchain-tavily python-dotenv
      streamlit run humanintheloop_ui.py
"""
import os
import uuid
from typing import TypedDict, Annotated

import streamlit as st
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

# ─────────────────────────────────────────────────────────────
# 1. LANGGRAPH BACKEND (your original logic, small UI tweaks)
# ─────────────────────────────────────────────────────────────
search_tool = TavilySearch(max_results=3)
tools = [search_tool]

writer_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)
writer_llm_with_tools = writer_llm.bind_tools(tools)   # writer may call Tavily


class State(TypedDict):
    topic: str
    messages: Annotated[list, add_messages]
    draft: str
    review_feedback: str
    is_approved: bool
    attempt: int
    max_attempts: int      # added: controlled by the UI slider (was hard-coded 3)


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
    """Writes / rewrites the post. May call Tavily first."""
    msgs = state["messages"]

    # Last message is a search result -> continue the SAME attempt.
    if msgs and isinstance(msgs[-1], ToolMessage):
        response = writer_llm_with_tools.invoke([("system", WRITER_SYSTEM_PROMPT)] + msgs)
        return {"messages": [response]}

    # Otherwise it is a fresh attempt.
    attempt = state.get("attempt", 0) + 1
    topic = state["topic"]
    if attempt == 1:
        user_message = (f"Write a LinkedIn post on this topic: {topic}. "
                        "If you need current info, search the web first.")
    else:
        user_message = (f"Your previous draft on '{topic}' was rejected. "
                        f"Here is the reviewer's feedback:\n\n{state.get('review_feedback', '')}\n\n"
                        "Write a new, improved draft that fixes every issue mentioned. "
                        "Do not repeat the mistake.")
    response = writer_llm_with_tools.invoke(
        [("system", WRITER_SYSTEM_PROMPT)] + msgs + [("human", user_message)]
    )
    return {"messages": [("human", user_message), response], "attempt": attempt}


tool_node = ToolNode(tools)   # runs the Tavily calls requested by the writer


def extract_draft_node(state: State) -> dict:
    """Final AI message (no tool calls) becomes the draft."""
    return {"draft": state["messages"][-1].content}


def human_review_node(state: State) -> dict:
    """PAUSES the graph. Resumes with whatever the human sends via Command(resume=...)."""
    human_response = interrupt({"draft": state["draft"], "attempt": state["attempt"]})
    text = human_response.strip()
    if text.lower() in ["approved", "approve", "yes", "ok", "good"]:
        return {"is_approved": True, "review_feedback": "Approved by human."}
    return {"is_approved": False, "review_feedback": text}


def should_use_tool(state: State):
    """Writer requested a search? -> tools, else extract the draft."""
    return "tools" if getattr(state["messages"][-1], "tool_calls", None) else "extract_draft"


def should_use_looping(state: State):
    """Approved or attempts exhausted -> END, else back to writer."""
    if state["is_approved"] or state["attempt"] >= state.get("max_attempts", 3):
        return END
    return "writer"


# cache_resource is IMPORTANT: Streamlit reruns the script on every click, and
# the MemorySaver (checkpointer) must survive reruns or the paused graph is lost.
@st.cache_resource
def build_app():
    g = StateGraph(State)
    g.add_node("writer", writer_node)
    g.add_node("tools", tool_node)
    g.add_node("extract_draft", extract_draft_node)
    g.add_node("human_review", human_review_node)
    g.add_edge(START, "writer")
    g.add_conditional_edges("writer", should_use_tool)
    g.add_edge("tools", "writer")
    g.add_edge("extract_draft", "human_review")
    g.add_conditional_edges("human_review", should_use_looping)
    return g.compile(checkpointer=MemorySaver())


# ─────────────────────────────────────────────────────────────
# 2. STYLING (animated gradient, glass cards, hover buttons)
# ─────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family:'Poppins',sans-serif; }

/* Moving gradient background */
.stApp { background:linear-gradient(-45deg,#0b132b,#1c2541,#3a0ca3,#560bad,#0f4c5c);
         background-size:400% 400%; animation:bgShift 18s ease infinite; }
@keyframes bgShift { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
@keyframes fadeDown { from{opacity:0;transform:translateY(-25px)} to{opacity:1;transform:none} }
@keyframes fadeUp   { from{opacity:0;transform:translateY(25px)}  to{opacity:1;transform:none} }
@keyframes shine    { to { background-position:300% center; } }

/* Shimmering title */
.hero { text-align:center; padding:1.5rem 0 .5rem; animation:fadeDown .9s ease; }
.hero h1 { font-size:2.8rem; font-weight:800; margin:0;
  background:linear-gradient(90deg,#22d3ee,#a78bfa,#f472b6,#facc15,#22d3ee);
  background-size:300% auto; -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  animation:shine 6s linear infinite; }
.hero p { color:#cbd5e1; }

/* Glass cards (green = approved, pink = your feedback, cyan = draft) */
.card { background:rgba(255,255,255,.07); backdrop-filter:blur(14px);
  border:1px solid rgba(255,255,255,.15); border-radius:18px; padding:1.1rem 1.4rem;
  margin:.8rem 0; color:#f1f5f9; animation:fadeUp .6s ease; transition:transform .25s, box-shadow .25s; }
.card:hover { transform:translateY(-4px); box-shadow:0 12px 30px rgba(167,139,250,.35); }
.card.ok  { border-left:6px solid #4ade80; }
.card.fb  { border-left:6px solid #f472b6; }
.card.draft { border-left:6px solid #22d3ee; white-space:pre-wrap; line-height:1.6; }

/* Pipeline tracker */
.pipe { display:flex; justify-content:center; gap:14px; flex-wrap:wrap; margin:1rem 0; }
.step { padding:.7rem 1.1rem; border-radius:14px; color:#94a3b8; font-weight:600;
  background:rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.12); transition:all .3s; }
.step.done   { color:#052e16; background:#4ade80; box-shadow:0 0 14px #4ade8088; }
.step.active { color:#fff; background:linear-gradient(135deg,#f472b6,#a78bfa);
               animation:pulse 1s ease-in-out infinite; }
@keyframes pulse { 0%,100%{transform:scale(1);box-shadow:0 0 0 0 #f472b688}
                   50%{transform:scale(1.08);box-shadow:0 0 22px 6px #a78bfa88} }

/* Buttons: gradient shift + lift + glow on hover, shrink on click */
div.stButton > button { width:100%; border:none; border-radius:14px; padding:.8rem 1rem;
  font-weight:700; font-size:1.05rem; color:#fff;
  background:linear-gradient(135deg,#22d3ee,#a78bfa,#f472b6); background-size:200% 200%;
  transition:transform .2s, box-shadow .2s, background-position .5s; }
div.stButton > button:hover  { transform:translateY(-3px) scale(1.03); background-position:100% 0;
  box-shadow:0 10px 28px rgba(167,139,250,.6); color:#fff; }
div.stButton > button:active { transform:scale(.96); }
label, .stMarkdown { color:#e2e8f0 !important; }
</style>
"""

# (emoji, label, node name) used by the tracker
STEPS = [("✍️", "Writer", "writer"), ("🔍", "Web Search", "tools"),
         ("📝", "Draft", "extract_draft"), ("🧑‍💻", "Your Review", "human_review")]


def pipeline_html(done: set, active) -> str:
    """grey = waiting, pulsing = active, green = done."""
    out = '<div class="pipe">'
    for emoji, label, node in STEPS:
        cls = "active" if node == active else ("done" if node in done else "")
        out += f'<div class="step {cls}">{emoji} {label}</div>'
    return out + "</div>"


# ─────────────────────────────────────────────────────────────
# 3. SESSION STATE + GRAPH RUNNER
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="LinkedIn Post Generator (HITL)", page_icon="🧑‍💻", layout="centered")
st.markdown(CSS, unsafe_allow_html=True)

ss = st.session_state
ss.setdefault("phase", "idle")                     # idle -> review -> done
ss.setdefault("thread_id", str(uuid.uuid4()))      # one checkpoint thread per browser session
ss.setdefault("gstate", {})                        # latest graph state
ss.setdefault("history", [])                       # your past decisions, shown as cards

app = build_app()
config = {"configurable": {"thread_id": ss.thread_id}, "recursion_limit": 40}


def run_graph(graph_input, tracker):
    """Streams the graph until it finishes OR pauses at the human_review interrupt."""
    done, active = set(), "writer"
    tracker.markdown(pipeline_html(done, active), unsafe_allow_html=True)

    for event in app.stream(graph_input, config=config, stream_mode="updates"):
        for node in event:
            if node == "__interrupt__":            # pause signal, not a real node
                continue
            done.add(node)
            active = "writer" if node == "tools" else None
            tracker.markdown(pipeline_html(done, active), unsafe_allow_html=True)

    # snapshot.next is non-empty when the graph is paused waiting for us
    snap = app.get_state(config)
    ss.gstate = snap.values
    ss.phase = "review" if snap.next else "done"


# ─────────────────────────────────────────────────────────────
# 4. PAGE
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="hero"><h1>🧑‍💻 LinkedIn Post Generator</h1>'
            '<p>AI writes. You decide. Approve it or send feedback until it is perfect.</p></div>',
            unsafe_allow_html=True)

missing = [k for k in ("GROQ_API_KEY", "TAVILY_API_KEY") if not os.getenv(k)]
if missing:
    st.error(f"Missing in .env: {', '.join(missing)}")

tracker = st.empty()   # placeholder for the animated pipeline

# ── PHASE: idle (ask for topic) ──
if ss.phase == "idle":
    topic = st.text_input("💡 Topic", placeholder="e.g. Why RAG beats fine-tuning for small teams")
    max_attempts = st.slider("🔁 Max attempts", 1, 5, 3)
    if st.button("✨ Generate First Draft", disabled=bool(missing)):
        if not topic.strip():
            st.warning("Pehle topic likhiye 🙂")
        else:
            initial = {"topic": topic.strip(), "messages": [], "draft": "", "review_feedback": "",
                       "is_approved": False, "attempt": 0, "max_attempts": max_attempts}
            with st.spinner("Writer draft bana raha hai..."):
                run_graph(initial, tracker)
            st.rerun()   # redraw the page in its new phase

# ── PHASE: review (graph is paused, waiting for YOU) ──
elif ss.phase == "review":
    v = ss.gstate
    tracker.markdown(pipeline_html({"writer", "tools", "extract_draft"}, "human_review"),
                     unsafe_allow_html=True)

    # Earlier feedback you gave (so you can see the improvement loop)
    for h in ss.history:
        st.markdown(f'<div class="card fb"><b>Attempt {h["attempt"]} - your feedback</b><br>{h["text"]}</div>',
                    unsafe_allow_html=True)

    st.markdown(f"### 📝 Draft (Attempt {v['attempt']} of {v['max_attempts']})")
    st.markdown(f'<div class="card draft">{v["draft"]}</div>', unsafe_allow_html=True)
    st.caption(f"{len(v['draft'].split())} words")

    feedback = st.text_area("💬 Feedback (only needed if you want changes)",
                            key=f"fb_{v['attempt']}",   # new key per attempt -> box clears itself
                            placeholder="e.g. Make the hook stronger and shorten it to ~150 words")
    c1, c2 = st.columns(2)

    if c1.button("✅ Approve"):
        # 'approved' is one of the keywords human_review_node understands
        with st.spinner("Finalising..."):
            run_graph(Command(resume="approved"), tracker)
        st.rerun()

    if c2.button("🔁 Request Changes"):
        if not feedback.strip():
            st.warning("Feedback likhiye, phir Request Changes dabaiye.")
        else:
            ss.history.append({"attempt": v["attempt"], "text": feedback.strip()})
            with st.spinner("Writer naya draft bana raha hai..."):
                run_graph(Command(resume=feedback.strip()), tracker)   # resumes the paused graph
            st.rerun()

# ── PHASE: done (approved or max attempts reached) ──
else:
    v = ss.gstate
    tracker.markdown(pipeline_html({s[2] for s in STEPS}, None), unsafe_allow_html=True)
    if v.get("is_approved"):
        st.balloons()
        st.success("Post approved! 🎉")
    else:
        st.warning("Max attempts reach ho gaye - last draft neeche hai.")

    st.markdown("### 📄 Final Post")
    st.code(v["draft"], language=None)          # built-in copy button
    m1, m2 = st.columns(2)
    m1.metric("Attempts", v["attempt"])
    m2.metric("Words", len(v["draft"].split()))

    if st.button("🆕 Write Another Post"):
        # new thread id = fresh checkpoint, old conversation is not reused
        for k in ("phase", "thread_id", "gstate", "history"):
            ss.pop(k, None)
        st.rerun()