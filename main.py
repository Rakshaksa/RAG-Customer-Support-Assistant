"""
RAG-Based Customer Support Assistant
Stack: LangChain + LangGraph + ChromaDB + HuggingFace + Groq
Phases: Load -> Chunk -> Embed -> Retrieve -> Route -> Answer / HITL
"""

import os
from typing import TypedDict, Literal

# PHASE 1 - Load PDF
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("custe.pdf")
docs = loader.load()
print(f"[Phase 1] Loaded {len(docs)} page(s)")

# PHASE 2 - Chunk
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20, separators=["\n\n", "\n", ". ", " ", ""])
chunks = splitter.split_documents(docs)
print(f"[Phase 2] Created {len(chunks)} chunk(s)")

# PHASE 3 - Embed + Store
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
print(f"[Phase 3] Stored {vectorstore._collection.count()} embeddings in ChromaDB")

# PHASE 4 - Retriever + LLM
from langchain_groq import ChatGroq
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.environ.get("GROQ_API_KEY", "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"))
print("[Phase 4] Retriever and LLM ready")

# PHASE 5 - State Definition
class State(TypedDict):
    question:    str
    context:     str
    answer:      str
    confidence:  str
    needs_human: bool

# PHASE 6 - Routing Logic
SENSITIVE_KEYWORDS = ["legal", "lawsuit", "sue", "fraud", "scam", "complaint", "police", "court", "refund denied", "manager", "supervisor", "cheated", "stolen"]
AMBIGUOUS_KEYWORDS = ["maybe", "not sure", "i think", "confused", "don't know", "unclear"]

def assess_confidence(context: str, question: str) -> str:
    if not context or len(context.strip()) < 20:
        return "none"
    q = question.lower()
    if any(kw in q for kw in SENSITIVE_KEYWORDS):
        return "low"
    if any(kw in q for kw in AMBIGUOUS_KEYWORDS):
        return "low"
    return "high"

def route_node(state: State) -> State:
    print("[Phase 6] Evaluating routing...")
    confidence = assess_confidence(state.get("context", ""), state.get("question", ""))
    needs_human = confidence in ("low", "none")
    print(f"          Confidence: {confidence} | Escalate: {needs_human}")
    return {"confidence": confidence, "needs_human": needs_human}

def should_escalate(state: State) -> Literal["answer", "hitl", "fallback"]:
    c = state.get("confidence", "none")
    if c == "none":   return "fallback"
    if c == "low":    return "hitl"
    return "answer"

# Nodes
def retrieve_node(state: State) -> State:
    print("[Node: retrieve] Searching knowledge base...")
    retrieved = retriever.invoke(state.get("question", ""))
    context = "\n".join([d.page_content for d in retrieved])
    print(f"          Retrieved {len(retrieved)} chunk(s)")
    return {"context": context}

def answer_node(state: State) -> State:
    print("[Node: answer] Generating LLM answer...")
    prompt = f"""You are a helpful customer support assistant.
Answer using ONLY the context below. If not found, say you don't have that information.

Context:
{state.get('context', '')}

Question: {state.get('question', '')}
Answer:"""
    response = llm.invoke(prompt)
    return {"answer": response.content}

def fallback_node(state: State) -> State:
    print("[Node: fallback] No relevant content found.")
    return {"answer": "I'm sorry, I couldn't find relevant information. Please contact support@shopease.com or call 1800-XXX-XXX."}

# PHASE 7 - HITL Escalation
def hitl_node(state: State) -> State:
    print("\n" + "="*50)
    print("  HITL ESCALATION TRIGGERED")
    print(f"  Reason   : Confidence = '{state.get('confidence')}'")
    print(f"  Question : {state.get('question', '')}")
    print("  Routing to human agent...")
    print("="*50)
    # Production: create Zendesk ticket, email agent, poll for response
    # Demo: CLI simulation
    human_response = input("\n[HUMAN AGENT] Enter response: ").strip()
    if not human_response:
        human_response = "A human agent will follow up with you shortly via email."
    return {"answer": f"[Support Agent]: {human_response}"}

# PHASE 5 - Build Graph
from langgraph.graph import StateGraph, END

graph = StateGraph(State)
graph.add_node("retrieve", retrieve_node)
graph.add_node("route",    route_node)
graph.add_node("answer",   answer_node)
graph.add_node("fallback", fallback_node)
graph.add_node("hitl",     hitl_node)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "route")
graph.add_conditional_edges("route", should_escalate, {"answer": "answer", "fallback": "fallback", "hitl": "hitl"})
graph.add_edge("answer",   END)
graph.add_edge("fallback", END)
graph.add_edge("hitl",     END)

app = graph.compile()
print("[Phase 5] LangGraph workflow compiled\n")

# Run
def run(question: str) -> str:
    result = app.invoke({"question": question, "context": "", "answer": "", "confidence": "high", "needs_human": False})
    return result.get("answer", "No answer generated.")

if __name__ == "__main__":
    print("="*50)
    print("  ShopEasy RAG Customer Support Assistant")
    print("  Type 'exit' to quit")
    print("="*50 + "\n")

    test_questions = [
        "How do I return a product?",
        "How long does a refund take?",
        "I want to file a legal complaint.",
        "What is the weather today?",
    ]

    for q in test_questions:
        print(f"\n{'─'*50}")
        print(f"QUESTION : {q}")
        print(f"ANSWER   : {run(q)}")

    print(f"\n{'─'*50}")
    print("Interactive mode (type 'exit' to quit)\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit", "q"):
            break
        if user_input:
            print(f"Bot: {run(user_input)}\n")