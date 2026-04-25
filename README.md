# 🤖 RAG-Based Customer Support Assistant
### Built with LangChain | LangGraph | ChromaDB | HuggingFace | Groq

---

## 📌 Project Overview

An AI-powered Customer Support Assistant that answers user queries
from a PDF knowledge base using Retrieval-Augmented Generation (RAG).

The system uses a graph-based workflow (LangGraph) with intelligent
routing logic and Human-in-the-Loop (HITL) escalation for sensitive
or unanswerable queries.

---

## 🎯 Features

- ✅ PDF ingestion and chunking pipeline
- ✅ Semantic search using HuggingFace embeddings + ChromaDB
- ✅ LLM-powered answer generation via Groq (LLaMA 3.1)
- ✅ LangGraph workflow with 4 nodes and conditional routing
- ✅ 3-way routing: Answer / Fallback / HITL Escalation
- ✅ Human-in-the-Loop escalation for sensitive queries
- ✅ Graceful fallback when no relevant content is found

---

## 🏗️ System Architecture

```
User Query
    │
    ▼
[Retrieve Node] ──► ChromaDB (Vector Search)
    │
    ▼
[Route Node] ──────► Confidence Assessment
    │
    ├──► HIGH  ──► [Answer Node]   ──► LLM (Groq) ──► Answer
    ├──► LOW   ──► [HITL Node]     ──► Human Agent ──► Answer
    └──► NONE  ──► [Fallback Node] ──► Default Response
```

---

## 🧩 Tech Stack

| Component        | Technology                        |
|------------------|-----------------------------------|
| Language         | Python 3.10+                      |
| PDF Loader       | LangChain PyPDFLoader             |
| Text Splitter    | RecursiveCharacterTextSplitter    |
| Embedding Model  | HuggingFace all-MiniLM-L6-v2     |
| Vector Database  | ChromaDB                          |
| LLM              | Groq — LLaMA 3.1 8B Instant      |
| Workflow Engine  | LangGraph StateGraph              |
| HITL             | CLI simulation (extensible)       |

---

## 📁 Folder Structure

```
RAG-Customer-Support-Assistant/
│
├── main.py            ← Complete pipeline (Phases 1–7)
├── create_pdf.py      ← Script to generate knowledge base PDF
├── custe.pdf          ← FAQ knowledge base document
├── requirements.txt   ← Python dependencies
├── .gitignore         ← Git ignore rules
└── README.md          ← Project documentation
```

---

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/RAG-Customer-Support-Assistant.git
cd RAG-Customer-Support-Assistant
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set your Groq API key**
```bash
# Windows
set GROQ_API_KEY=your_groq_api_key_here

# Mac/Linux
export GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at: https://console.groq.com

---

## ▶️ How to Run

```bash
python main.py
```

---

## 💬 Sample Input / Output

| Query | Route | Output |
|-------|-------|--------|
| How do I return a product? | Answer Node | You can return within 30 days with receipt |
| How long does a refund take? | Answer Node | Refunds processed in 5-7 business days |
| I want to file a legal complaint | HITL Node | Escalated to human agent |
| What is the weather today? | Fallback Node | Contact support@shopease.com |

---

## 🔀 Routing Logic

```python
HIGH confidence  →  Answer Node   (context found + safe query)
LOW confidence   →  HITL Node     (sensitive/ambiguous query)
NO context       →  Fallback Node (nothing retrieved)
```

**Sensitive keywords that trigger HITL:**
`legal, lawsuit, fraud, complaint, manager, scam, court, stolen`

---
