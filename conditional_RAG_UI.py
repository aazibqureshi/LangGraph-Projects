"""
Conditional RAG College Assistant - Colourful Streamlit Chat UI
LangGraph flow: Classifier -> (Academic PDF | Fee PDF | No retrieval) -> Response

Run:  pip install streamlit langgraph langchain-groq langchain-community \
                  langchain-huggingface langchain-text-splitters faiss-cpu pypdf \
                  sentence-transformers python-dotenv
      streamlit run conditional_rag_ui.py
Keep academics_handbook.pdf and fee_structure.pdf in the same folder.
"""
import os
from typing import TypedDict, Annotated

import streamlit as st
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

ACADEMIC_PDF = "academics_handbook.pdf"
FEE_PDF = "fee_structure.pdf"
NO_RETRIEVAL = "NO_RETRIEVAL_NEEDED"   # sentinel: general chat needs no PDF


# ─────────────────────────────────────────────────────────────
# 1. LANGGRAPH BACKEND (your original logic)
# ─────────────────────────────────────────────────────────────
class State(TypedDict):
    programme: str                            # BCA / BBA / BCOM (H)
    messages: Annotated[list, add_messages]   # conversation (auto-appended)
    query_type: str                           # 'academic' | 'fee' | 'general'
    retrieved_context: str                    # text pulled from the PDF


# cache_resource: PDFs are embedded and the graph compiled ONCE, not on every
# Streamlit rerun (which happens on every click / message).
@st.cache_resource(show_spinner="📚 Loading PDFs and building indexes...")
def build_app():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def build_retriever(pdf_path: str):
        """PDF -> overlapping chunks -> FAISS index -> top-4 retriever."""
        docs = PyPDFLoader(pdf_path).load()
        chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(docs)
        return FAISS.from_documents(chunks, embeddings).as_retriever(search_kwargs={"k": 4})

    academic_retriever = build_retriever(ACADEMIC_PDF)
    fee_retriever = build_retriever(FEE_PDF)
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.4)

    def classifier_node(state: State) -> dict:
        """Ask the LLM to label the query: academic / fee / general."""
        q = state["messages"][-1].content
        prompt = (
            "Classify the following student query into exactly one category: "
            "'academic', 'fee', or 'general'.\n\n"
            "Use 'academic' for questions about attendance, exams, grading, credits, "
            "promotions, course structure, summer training or degree requirements.\n"
            "Use 'fee' for questions about tuition, payment, refund, late charges, "
            "scholarships, or any money-related topic.\n"
            "Use 'general' for greetings, casual talk, or anything not related to "
            "the college rules or fee.\n\n"
            f"Query: {q}\n\nReturn only one word: academic, fee, or general."
        )
        label = llm.invoke(prompt).content.strip().lower()
        # Defensive parsing: LLM may add extra words.
        label = "academic" if "academic" in label else "fee" if "fee" in label else "general"
        return {"query_type": label}

    def academic_rag_node(state: State) -> dict:
        docs = academic_retriever.invoke(state["messages"][-1].content)
        return {"retrieved_context": "\n\n".join(d.page_content for d in docs)}

    def fee_rag_node(state: State) -> dict:
        docs = fee_retriever.invoke(state["messages"][-1].content)
        return {"retrieved_context": "\n\n".join(d.page_content for d in docs)}

    def general_node(state: State) -> dict:
        return {"retrieved_context": NO_RETRIEVAL}

    def response_node(state: State) -> dict:
        """Final answer, grounded in PDF context when available."""
        q = state["messages"][-1].content
        prog = state.get("programme", "Unknown")
        ctx = state["retrieved_context"]
        if ctx == NO_RETRIEVAL:
            prompt = (f"You are a friendly college assistant talking to a {prog} student. "
                      f"Answer this question using your own general knowledge:\n\n{q}")
        else:
            prompt = (f"You are a college assistant helping a {prog} student. "
                      "Use the following context from the official college document to answer "
                      "the question accurately. If the context mentions specific figures for "
                      f"different programmes, highlight the one relevant to {prog} if possible.\n\n"
                      f"Context:\n{ctx}\n\nQuestion: {q}\n\n"
                      "Give a clear, friendly and precise answer.")
        return {"messages": [("ai", llm.invoke(prompt).content.strip())]}

    def route_query(state: State):
        """Conditional edge: pick the branch from the classifier's label."""
        return {"academic": "academic_rag", "fee": "fee_rag"}.get(state["query_type"], "general")

    g = StateGraph(State)
    g.add_node("classifier", classifier_node)
    g.add_node("academic_rag", academic_rag_node)
    g.add_node("fee_rag", fee_rag_node)
    g.add_node("general", general_node)
    g.add_node("response", response_node)
    g.add_edge(START, "classifier")
    # Explicit path map = safer than relying on return-type inference.
    g.add_conditional_edges("classifier", route_query,
                            {"academic_rag": "academic_rag", "fee_rag": "fee_rag", "general": "general"})
    for branch in ("academic_rag", "fee_rag", "general"):
        g.add_edge(branch, "response")      # all branches converge here
    g.add_edge("response", END)
    return g.compile()


# ─────────────────────────────────────────────────────────────
# 2. STYLING (animated gradient, glass chat bubbles, hover buttons)
# ─────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family:'Poppins',sans-serif; }
.stApp { background:linear-gradient(-45deg,#0f172a,#1e1b4b,#134e4a,#4c1d95,#831843);
         background-size:400% 400%; animation:bgShift 20s ease infinite; }
@keyframes bgShift { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
@keyframes fadeDown { from{opacity:0;transform:translateY(-25px)} to{opacity:1;transform:none} }
@keyframes fadeUp   { from{opacity:0;transform:translateY(18px)}  to{opacity:1;transform:none} }
@keyframes shine    { to { background-position:300% center; } }

/* Title */
.hero { text-align:center; padding:1rem 0 .3rem; animation:fadeDown .9s ease; }
.hero h1 { font-size:2.6rem; font-weight:800; margin:0;
  background:linear-gradient(90deg,#34d399,#60a5fa,#f472b6,#fbbf24,#34d399); background-size:300% auto;
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; animation:shine 6s linear infinite; }
.hero p { color:#cbd5e1; margin:.2rem 0 0; }

/* Chat bubbles */
[data-testid="stChatMessage"] { background:rgba(255,255,255,.07); backdrop-filter:blur(12px);
  border:1px solid rgba(255,255,255,.14); border-radius:18px; animation:fadeUp .5s ease;
  transition:transform .25s, box-shadow .25s; color:#f1f5f9; }
[data-testid="stChatMessage"]:hover { transform:translateY(-2px); box-shadow:0 10px 26px rgba(96,165,250,.3); }
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li { color:#f1f5f9; }

/* Route badges (which branch answered) */
.badge { display:inline-block; padding:.2rem .8rem; border-radius:999px; font-size:.78rem;
  font-weight:700; color:#fff; margin-bottom:.4rem; }
.badge.academic { background:linear-gradient(135deg,#6366f1,#22d3ee); }
.badge.fee      { background:linear-gradient(135deg,#f59e0b,#ef4444); }
.badge.general  { background:linear-gradient(135deg,#10b981,#84cc16); }

/* Pipeline tracker */
.pipe { display:flex; gap:10px; flex-wrap:wrap; margin:.2rem 0 .6rem; }
.step { padding:.45rem .9rem; border-radius:12px; color:#94a3b8; font-weight:600; font-size:.85rem;
  background:rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.12); transition:all .3s; }
.step.done   { color:#052e16; background:#4ade80; box-shadow:0 0 12px #4ade8088; }
.step.active { color:#fff; background:linear-gradient(135deg,#f472b6,#818cf8);
               animation:pulse 1s ease-in-out infinite; }
@keyframes pulse { 0%,100%{transform:scale(1)} 50%{transform:scale(1.08); box-shadow:0 0 20px 4px #818cf888} }

/* Buttons: gradient shift + lift + glow on hover, shrink on click */
div.stButton > button { width:100%; border:none; border-radius:14px; padding:.65rem .8rem;
  font-weight:600; color:#fff; background:linear-gradient(135deg,#34d399,#60a5fa,#f472b6);
  background-size:200% 200%; transition:transform .2s, box-shadow .2s, background-position .5s; }
div.stButton > button:hover  { transform:translateY(-3px) scale(1.03); background-position:100% 0;
  box-shadow:0 10px 26px rgba(96,165,250,.6); color:#fff; }
div.stButton > button:active { transform:scale(.96); }
label, .stMarkdown, .stRadio { color:#e2e8f0 !important; }

/* ---- Full-page gradient: remove Streamlit's default dark header / bottom bars ---- */
.stApp { background-attachment: fixed; }                       /* gradient covers whole viewport */
[data-testid="stHeader"] { background: transparent !important; }          /* top bar (Deploy menu) */
[data-testid="stBottom"], [data-testid="stBottom"] > div,
[data-testid="stBottomBlockContainer"] { background: transparent !important; }  /* bottom chat area */
[data-testid="stChatInput"] { background: rgba(255,255,255,.14) !important; backdrop-filter: blur(12px);
  border: 1px solid rgba(255,255,255,.25); border-radius: 16px; transition: box-shadow .25s; }
[data-testid="stChatInput"]:focus-within { box-shadow: 0 0 22px rgba(96,165,250,.6); }
[data-testid="stChatInput"] textarea { color: #fff !important; }
</style>
"""

BADGES = {"academic": "🎓 Academic PDF", "fee": "💰 Fee PDF", "general": "💬 General Chat"}
RETRIEVE_LABEL = {"academic": "🎓 Academic PDF", "fee": "💰 Fee PDF", "general": "💬 No Retrieval"}
EXAMPLES = ["What is the minimum attendance required?", "What are the tuition fees?",
            "Is there a refund policy?", "Hi! What can you help me with?"]


def pipeline_html(done: set, active, route) -> str:
    """Tracker: Classifier -> Retriever (label changes with route) -> Response."""
    steps = [("classifier", "🧭 Classifier"),
             ("retrieve", RETRIEVE_LABEL.get(route, "📚 Retriever")),
             ("response", "✨ Response")]
    out = '<div class="pipe">'
    for key, label in steps:
        cls = "active" if key == active else ("done" if key in done else "")
        out += f'<div class="step {cls}">{label}</div>'
    return out + "</div>"


def render_assistant(content: str, route: str, context: str):
    """One assistant bubble: route badge + answer + optional source expander."""
    st.markdown(f'<span class="badge {route}">{BADGES.get(route, route)}</span>', unsafe_allow_html=True)
    st.markdown(content)
    if context and context != NO_RETRIEVAL:
        with st.expander("📄 Source context used"):
            st.text(context)


# ─────────────────────────────────────────────────────────────
# 3. PAGE
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="College RAG Assistant", page_icon="🎓", layout="centered")
st.markdown(CSS, unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🎓 College Assistant</h1>'
            '<p>Ask about academics or fees. The graph routes your question to the right PDF.</p></div>',
            unsafe_allow_html=True)

# Fail early with a clear message instead of a traceback.
if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY missing in .env")
    st.stop()
missing_pdfs = [p for p in (ACADEMIC_PDF, FEE_PDF) if not os.path.exists(p)]
if missing_pdfs:
    st.error(f"PDF not found: {', '.join(missing_pdfs)} (keep them next to this script)")
    st.stop()

app = build_app()

ss = st.session_state
ss.setdefault("chat", [])      # NOTE: avoid keys named like dict methods (values, keys, items...)

programme = st.radio("🎒 Your programme", ["BCA", "BBA", "BCOM (H)"], horizontal=True)

# Quick-question buttons: clicking queues the text, handled below like typed input.
st.caption("Try a quick question:")
cols = st.columns(2)
for i, q in enumerate(EXAMPLES):
    if cols[i % 2].button(q, key=f"ex_{i}"):
        ss.queued = q

if ss.chat and st.button("🗑️ Clear chat"):
    ss.chat = []
    st.rerun()

# Replay history (rerun-safe: Streamlit redraws everything each interaction).
for m in ss.chat:
    with st.chat_message(m["role"], avatar="🧑‍🎓" if m["role"] == "user" else "🤖"):
        if m["role"] == "user":
            st.markdown(m["content"])
        else:
            render_assistant(m["content"], m["route"], m["context"])

# New input: typed, or queued from an example button.
prompt = st.chat_input("Ask about attendance, exams, fees...") or ss.pop("queued", None)

if prompt:
    ss.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        tracker = st.empty()
        route, context, answer = None, "", ""
        done, active = set(), "classifier"
        tracker.markdown(pipeline_html(done, active, route), unsafe_allow_html=True)

        # stream_mode="updates" -> one event as each node finishes
        for event in app.stream({"programme": programme, "messages": [("human", prompt)]},
                                stream_mode="updates"):
            for node, update in event.items():
                if node == "classifier":
                    route = update["query_type"]
                    done.add("classifier"); active = "retrieve"
                elif node in ("academic_rag", "fee_rag", "general"):
                    context = update["retrieved_context"]
                    done.add("retrieve"); active = "response"
                elif node == "response":
                    # The update can hold the raw ("ai", text) tuple or an AIMessage - handle both.
                    msg = update["messages"][-1]
                    answer = msg[1] if isinstance(msg, tuple) else msg.content
                    done.add("response"); active = None
                tracker.markdown(pipeline_html(done, active, route), unsafe_allow_html=True)

        tracker.empty()   # hide tracker once done; badge shows the route instead
        render_assistant(answer, route, context)

    ss.chat.append({"role": "assistant", "content": answer, "route": route, "context": context})