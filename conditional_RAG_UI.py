"""
Streamlit UI for the Conditional RAG College Assistant.

This file is completely independent of conditional_RAG.py (your original
script is left untouched). It re-implements the same LangGraph pipeline
(classifier -> academic/fee/general -> response) but wraps it in a
Streamlit chat interface instead of a terminal input() loop.

Run with:
    streamlit run streamlit_app.py
"""

import streamlit as st
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Page config + light styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="College Assistant",
    page_icon="🎓",
    layout="centered",
)

st.markdown(
    """
    <style>
    .stChatMessage { border-radius: 14px; }
    .app-title { font-size: 2rem; font-weight: 700; margin-bottom: 0; }
    .app-subtitle { color: #6b7280; margin-top: 0; margin-bottom: 1.2rem; }
    .badge {
        display: inline-block; padding: 3px 10px; border-radius: 999px;
        font-size: 0.75rem; font-weight: 600; margin-left: 8px;
    }
    .badge-academic { background: #dbeafe; color: #1e40af; }
    .badge-fee { background: #fef3c7; color: #92400e; }
    .badge-general { background: #dcfce7; color: #166534; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cached resources: retrievers + LLM (built once, reused across reruns)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading knowledge base...")
def build_retriever(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    document = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(document)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})


@st.cache_resource(show_spinner="Compiling assistant graph...")
def build_graph():
    academic_retriever = build_retriever("academics_handbook.pdf")
    fee_retriever = build_retriever("fee_structure.pdf")
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.4)

    class State(TypedDict):
        programme: str
        messages: Annotated[list, add_messages]
        query_type: str
        retrieved_context: str

    def classifier_node(state: State) -> dict:
        last_message = state["messages"][-1].content
        prompt = (
            "Classify the following student query into exactly one category: "
            "'academic', 'fee', or 'general'.\n\n"
            "Use 'academic' for questions about attendance, exams, grading, credits, "
            "promotions, course structure, summer training or degree requirements.\n"
            "Use 'fee' for questions about tuition, payment, refund, late charges, "
            "scholarships, or any money-related topic.\n"
            "Use 'general' for greetings, casual talk, or anything not related to "
            "the college rules or fee.\n\n"
            f"Query: {last_message}\n\n"
            "Return only one word: academic, fee, or general."
        )
        response = llm.invoke(prompt)
        category = response.content.strip().lower()
        if "academic" in category:
            category = "academic"
        elif "fee" in category:
            category = "fee"
        else:
            category = "general"
        return {"query_type": category}

    def academic_rag_node(state: State) -> dict:
        query = state["messages"][-1].content
        docs = academic_retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        return {"retrieved_context": context}

    def fee_rag_node(state: State) -> dict:
        query = state["messages"][-1].content
        docs = fee_retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        return {"retrieved_context": context}

    def general_node(state: State) -> dict:
        return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}

    def response_node(state: State) -> dict:
        query = state["messages"][-1].content
        programme = state.get("programme", "Unknown")
        context = state["retrieved_context"]

        if context == "NO_RETRIEVAL_NEEDED":
            prompt = (
                f"You are a friendly college assistant talking to a {programme} student. "
                f"Answer this question using your own general knowledge:\n\n{query}"
            )
        else:
            prompt = (
                f"You are a college assistant helping a {programme} student. "
                f"Use the following context from the official college document to answer "
                f"the question accurately. If the context mentions specific figures for "
                f"different programmes, highlight the one relevant to {programme} if possible.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                f"Give a clear, friendly and precise answer."
            )
        response = llm.invoke(prompt)
        return {"messages": [("ai", response.content.strip())]}

    def route_query(state: State):
        if state["query_type"] == "academic":
            return "academic_rag"
        elif state["query_type"] == "fee":
            return "fee_rag"
        else:
            return "general"

    graph = StateGraph(State)
    graph.add_node("classifier", classifier_node)
    graph.add_node("academic_rag", academic_rag_node)
    graph.add_node("fee_rag", fee_rag_node)
    graph.add_node("general", general_node)
    graph.add_node("response", response_node)

    graph.add_edge(START, "classifier")
    graph.add_conditional_edges("classifier", route_query)
    graph.add_edge("academic_rag", "response")
    graph.add_edge("fee_rag", "response")
    graph.add_edge("general", "response")
    graph.add_edge("response", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Sidebar: programme selector + chat controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🎓 Student Details")
    programme = st.selectbox(
        "Select your programme",
        options=["BCA", "BBA", "BCOM (H)"],
        index=0,
    )

    st.divider()
    st.markdown("### ℹ️ About")
    st.caption(
        "Ask about academics (attendance, exams, credits), fees "
        "(tuition, refunds, scholarships), or just say hi!"
    )

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main chat UI
# ---------------------------------------------------------------------------

st.markdown('<p class="app-title">College Assistant 🎓</p>', unsafe_allow_html=True)
st.markdown(
    f'<p class="app-subtitle">Chatting as a <b>{programme}</b> student</p>',
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    avatar = "🧑‍🎓" if msg["role"] == "human" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("category"):
            badge_class = f"badge badge-{msg['category']}"
            st.markdown(
                f'<span class="{badge_class}">{msg["category"]}</span>',
                unsafe_allow_html=True,
            )

# Chat input
user_query = st.chat_input("Ask about attendance, fees, exams...")

if user_query:
    st.session_state.messages.append({"role": "human", "content": user_query})
    with st.chat_message("human", avatar="🧑‍🎓"):
        st.markdown(user_query)

    with st.chat_message("ai", avatar="🤖"):
        with st.spinner("Thinking..."):
            app = build_graph()
            result = app.invoke({
                "programme": programme,
                "messages": [("human", user_query)],
            })
            answer = result["messages"][-1].content
            category = result.get("query_type", "general")

        st.markdown(answer)
        badge_class = f"badge badge-{category}"
        st.markdown(f'<span class="{badge_class}">{category}</span>', unsafe_allow_html=True)

    st.session_state.messages.append(
        {"role": "ai", "content": answer, "category": category}
    )