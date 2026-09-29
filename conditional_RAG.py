"""
Conditional RAG College Assistant
----------------------------------
This script builds a LangGraph pipeline that:
1. Classifies a student's query as 'academic', 'fee', or 'general'.
2. Retrieves relevant context from the matching PDF (if needed) using FAISS + HuggingFace embeddings.
3. Generates a final, programme-personalized answer using a Groq-hosted LLM.
"""

import os
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables (e.g. GROQ_API_KEY) from a .env file
load_dotenv()

# ---------------------------------------------------------------------------
# Step 1 - Building the RAG retrievers
# ---------------------------------------------------------------------------

# Embedding model used to convert text chunks into vectors for similarity search.
# all-MiniLM-L6-v2 is small, fast, and good enough for semantic search tasks like this.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_retriever(pdf_path: str):
    """
    Loads a PDF, splits it into overlapping text chunks, embeds those chunks,
    stores them in a FAISS vector index, and returns a retriever object
    that can fetch the top-k most relevant chunks for a given query.
    """
    loader = PyPDFLoader(pdf_path)
    document = loader.load()

    # Split the document into ~800-character chunks with 100-character overlap
    # so context isn't lost at chunk boundaries.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(document)

    # Build an in-memory FAISS vector store from the chunks.
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Return a retriever that fetches the top 4 most similar chunks per query.
    return vectorstore.as_retriever(search_kwargs={"k": 4})


# One retriever per knowledge source (academic rules vs fee structure).
academic_retriever = build_retriever("academics_handbook.pdf")
fee_retriever = build_retriever("fee_structure.pdf")

# The LLM used for both classification and final answer generation.
# temperature=0.4 keeps answers fairly factual but not too robotic.
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.4)

# ---------------------------------------------------------------------------
# Step 2 - Graph State
# ---------------------------------------------------------------------------
# This is the shared "memory" that flows between nodes in the graph.

class State(TypedDict):
    programme: str                              # Student's programme (BCA/BBA/BCOM)
    messages: Annotated[list, add_messages]      # Conversation history (auto-appended)
    query_type: str                              # 'academic' | 'fee' | 'general'
    retrieved_context: str                       # Context pulled from the relevant PDF

# ---------------------------------------------------------------------------
# Step 3 - Nodes (each node is a function that reads/updates the State)
# ---------------------------------------------------------------------------

def classifier_node(state: State) -> dict:
    """
    Looks at the latest user message and asks the LLM to classify it into
    one of three categories: academic, fee, or general. This decides which
    branch of the graph runs next.
    """
    last_message = state['messages'][-1].content

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

    # Defensive parsing: even if the LLM adds extra words, we only care
    # whether 'academic' or 'fee' appears in the reply; anything else -> general.
    if "academic" in category:
        category = "academic"
    elif "fee" in category:
        category = "fee"
    else:
        category = "general"

    return {"query_type": category}


def academic_rag_node(state: State) -> dict:
    """Retrieves the most relevant chunks from the academics handbook PDF."""
    query = state['messages'][-1].content
    docs = academic_retriever.invoke(query)
    # Join all retrieved chunk texts into one context block, separated by blank lines.
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}


def fee_rag_node(state: State) -> dict:
    """Retrieves the most relevant chunks from the fee structure PDF."""
    query = state["messages"][-1].content
    docs = fee_retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}


def general_node(state: State) -> dict:
    """
    For casual/greeting queries, no PDF lookup is needed.
    A sentinel value tells response_node to skip using retrieved context.
    """
    return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}


def response_node(state: State) -> dict:
    """
    Generates the final answer shown to the student.
    Uses retrieved PDF context when available, otherwise falls back to the
    LLM's own general knowledge. Always personalizes the tone using the
    student's programme (BCA/BBA/BCOM).
    """
    query = state["messages"][-1].content
    programme = state.get("programme", "Unknown")
    context = state["retrieved_context"]

    if context == "NO_RETRIEVAL_NEEDED":
        # General/casual query -> answer directly without any document context.
        prompt = (
            f"You are a friendly college assistant talking to a {programme} student. "
            f"Answer this question using your own general knowledge:\n\n{query}"
        )
    else:
        # Academic/fee query -> ground the answer in the retrieved PDF context.
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
    # ("ai", text) is LangChain's shorthand tuple format for an AIMessage;
    # add_messages will append it to the conversation history automatically.
    return {"messages": [("ai", response.content.strip())]}

# ---------------------------------------------------------------------------
# Step 4 - Router function (decides which node runs after the classifier)
# ---------------------------------------------------------------------------

def route_query(state: State):
    """Maps the classified query_type to the name of the next node to run."""
    if state['query_type'] == 'academic':
        return "academic_rag"
    elif state["query_type"] == "fee":
        return "fee_rag"
    else:
        return "general"

# ---------------------------------------------------------------------------
# Step 5 - Building the graph
# ---------------------------------------------------------------------------
# IMPORTANT: node names used in add_node() must exactly match the strings
# used in add_edge() / add_conditional_edges() and the values returned by
# route_query() - any mismatch (typo, hyphen vs underscore) breaks compile().

graph = StateGraph(State)

graph.add_node("classifier", classifier_node)
graph.add_node("academic_rag", academic_rag_node)
graph.add_node("fee_rag", fee_rag_node)
graph.add_node("general", general_node)
graph.add_node("response", response_node)

# Entry point: every run starts at the classifier node.
graph.add_edge(START, "classifier")

# Branching edge: after classification, route_query() decides the next node.
graph.add_conditional_edges(
    "classifier", route_query
)

# All three branches converge back into the response node.
graph.add_edge("academic_rag", "response")
graph.add_edge("fee_rag", "response")
graph.add_edge("general", "response")

# After generating the response, the graph run ends.
graph.add_edge("response", END)

# Compile the graph into a runnable app.
app = graph.compile()

# ---------------------------------------------------------------------------
# Step 6 - CLI loop to chat with the assistant
# ---------------------------------------------------------------------------

print("Welcome to the college Assistant\n\n")

print("Which programme are you in?")
print("1. BCA")
print("2. BBA")
print("3. BCOM (H)")

choice = input("\nEnter 1, 2 or 3: ")
programme_map = {
    "1": "BCA",
    "2": "BBA",
    "3": "BCOM (H)"
}

# Defaults to BCA if the user enters an invalid choice.
student_programme = programme_map.get(choice, "BCA")

print(f"\nGreat! You're set as a {student_programme} student.")

# Main chat loop: keeps invoking the graph for each new user query
# until the user types 'exit' or 'quit'.
while True:
    user_query = input("You: ")

    if user_query.lower() in ["exit", "quit"]:
        break

    result = app.invoke({
        "programme": student_programme,
        "messages": [("human", user_query)]
    })
    print(f"Assistant : {result['messages'][-1].content}")