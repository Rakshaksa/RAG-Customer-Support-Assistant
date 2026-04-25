# RAG-Based Customer Support Assistant

A production-ready Retrieval-Augmented Generation (RAG) system with Human-in-the-Loop (HITL) escalation, built with LangChain, LangGraph, ChromaDB, and Groq LLM.

## Features

✨ **RAG Pipeline**: PDF loading → Chunking → Embedding → Vector search
🧠 **Smart Routing**: Intent classification + confidence scoring + dynamic escalation
📊 **LangGraph Workflow**: Multi-step orchestration with conditional routing
🤝 **Human-in-the-Loop**: Easy escalation to human reviewers
⚡ **Optimized**: Chunk size 500, Top-K retrieval 3, Fast embeddings
📚 **Production Ready**: Modular design, comprehensive error handling, detailed documentation

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Your PDF
Place your customer support knowledge base as `custe.pdf` in the project root.

### 3. Run Simple RAG
```bash
python rag.py
```

### 4. Run Advanced Graph Workflow (with HITL)
```bash
python graph.py
```

Or use the improved entry point:
```bash
python main.py
```

## Architecture

### Simple RAG (`rag.py`)
- Direct PDF → Embedding → Retrieval → LLM → Answer
- Fast, straightforward responses
- Best for simple queries

### Advanced Workflow (`graph.py`)
```
Query → Intent Classification → Context Retrieval → Answer Generation
                                                          ↓
                                                  Confidence Evaluation
                                                          ↓
                                      ┌─────────────────┴──────────────┐
                                      ↓                                ↓
                          (High Confidence)                    (Low Confidence)
                                      ↓                                ↓
                                    Return                         HITL Escalation
                                   Answer                                ↓
                                                            Human Review & Response
                                                                        ↓
                                                                   Return Answer
```

## Project Structure

```
RAG proj/
├── config.py                 # Configuration parameters
├── document_processor.py      # PDF loading & chunking
├── llm_handler.py           # LLM interaction management
├── utils.py                 # Helper utilities
├── rag.py                   # Simple RAG application
├── graph.py                 # Advanced LangGraph workflow
├── main.py                  # Unified entry point
├── HLD.md                   # High-Level Design documentation
├── LLD.md                   # Low-Level Design documentation
├── custe.pdf                # Knowledge base (your PDF)
├── chroma_db/               # Vector database (auto-created)
├── venv/                    # Virtual environment
└── requirements.txt         # Python dependencies
```

## Configuration

Edit `config.py` to customize:

```python
# Document Processing
CHUNK_SIZE = 500          # Tokens per chunk
CHUNK_OVERLAP = 100       # Overlap for context preservation

# Model Selection
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"

# Retrieval
TOP_K_RETRIEVAL = 3       # Number of documents to retrieve
CONFIDENCE_THRESHOLD = 0.6 # Escalation threshold

# Persistence
CHROMA_DB_PATH = "./chroma_db"
```

## Key Components

### DocumentProcessor
Handles PDF processing:
- Loads PDF using PyPDFLoader
- Splits into optimal-sized chunks
- Generates embeddings
- Stores in ChromaDB

```python
from document_processor import DocumentProcessor

processor = DocumentProcessor()
vectorstore, retriever = processor.load_and_process_pdf("custe.pdf")
```

### LLMHandler
Manages LLM interactions:
- Q&A prompts
- Intent classification
- Confidence scoring
- Escalation decisions

```python
from llm_handler import LLMHandler

handler = LLMHandler()
qa_prompt = handler.create_qa_prompt()
intent_prompt = handler.create_intent_prompt()
```

### CustomerSupportGraph
LangGraph workflow orchestration:
- Multi-step reasoning
- Conditional routing
- HITL escalation
- State management

```python
from graph import CustomerSupportGraph

support_graph = CustomerSupportGraph()
result = support_graph.run("How do I return a product?")
print(result['answer'])
```

## Usage Examples

### Example 1: Simple Query
```python
from rag import RAGApplication

app = RAGApplication()
answer = app.answer("How do I track my order?")
print(answer)
```

### Example 2: With Escalation Support
```python
from graph import CustomerSupportGraph

support = CustomerSupportGraph()
result = support.run("I want a refund")

# Will escalate if confidence is low
if result['needs_escalation']:
    print(f"Escalation reason: {result['escalation_reason']}")
else:
    print(f"Answer: {result['answer']}")
```

### Example 3: Batch Processing
```python
from rag import RAGApplication

app = RAGApplication()
queries = [
    "How long does shipping take?",
    "What's your return policy?",
    "Can I change my order?"
]

for query in queries:
    answer = app.answer(query)
    print(f"Q: {query}\nA: {answer}\n")
```

## Performance Optimizations

### 1. Retrieval
- **Top-K = 3**: Balance between context and speed
- **Chunk Size = 500**: Optimal for context windows
- **Overlap = 100**: Preserves semantic continuity

### 2. Embedding
- **HuggingFace all-MiniLM-L6-v2**: Fast, lightweight, effective
- **Lazy Loading**: Embeddings loaded only when needed
- **Persistence**: Avoid re-embedding on restart

### 3. LLM
- **Temperature = 0.3**: Consistent, focused responses
- **Smaller Model**: Llama 3.1 8B (fast inference)
- **Streaming**: Possible addition for real-time responses

### 4. Workflow
- **Early Exit**: High-confidence queries skip HITL
- **Parallel Processing**: Potential for async retrieval
- **Caching**: Frequent questions can be cached

## Evaluation Metrics

### Quality Metrics
- **Exact Match**: Answer matches expected response (%)
- **BLEU Score**: Semantic similarity to reference answers
- **Human Judgment**: HITL review ratings

### Performance Metrics
- **Response Time**: Average end-to-end latency (seconds)
- **Token Usage**: Tokens per query (cost optimization)
- **Escalation Rate**: % of queries escalated to humans
- **Cache Hit Rate**: % of queries answered from cache

## Common Issues & Solutions

### Issue: Low Confidence Scores
**Solution**: 
- Improve PDF quality
- Increase chunk size to preserve context
- Tune confidence threshold in config.py

### Issue: Slow Retrieval
**Solution**:
- Reduce TOP_K_RETRIEVAL
- Use smaller embedding models
- Enable caching layer

### Issue: Wrong Intent Classification
**Solution**:
- Improve prompt engineering in llm_handler.py
- Add more examples to intent prompt
- Use few-shot learning

### Issue: Vector Database Not Found
**Solution**:
```bash
# Re-initialize the vector database
python -c "from document_processor import DocumentProcessor; DocumentProcessor().load_and_process_pdf('custe.pdf')"
```

## Development

### Adding New Features

1. **New Prompt**: Add to `LLMHandler.create_*_prompt()`
2. **New Node**: Add to `CustomerSupportGraph._build_graph()`
3. **New Config**: Add to `config.py`

### Testing
```bash
# Test RAG pipeline
python rag.py

# Test graph workflow
python graph.py

# Test specific components
python -c "from document_processor import DocumentProcessor; p = DocumentProcessor(); print(p.get_retriever())"
```

## Documentation

- **HLD.md**: System architecture, design decisions, scalability
- **LLD.md**: Implementation details, data structures, algorithms
- **README.md**: This file

## Requirements

- Python 3.8+
- 8GB RAM minimum
- GPU optional (for faster inference)

See `requirements.txt` for detailed dependencies.

## API Keys

Required API key for Groq LLM (get free from https://console.groq.com):

```python
# In config.py or as environment variable
LLM_API_KEY = "your-groq-api-key"
```

## Future Enhancements

- [ ] Multi-document RAG (multiple PDFs)
- [ ] Feedback loop for continuous improvement
- [ ] Caching layer for frequent queries
- [ ] Async/parallel processing
- [ ] Web UI dashboard
- [ ] Cost analytics and optimization
- [ ] A/B testing framework
- [ ] Fine-tuned domain models

## License

MIT License - Feel free to use this project for commercial or personal use.

## Support

For issues or questions:
1. Check the documentation (HLD.md, LLD.md)
2. Review error handling in relevant modules
3. Check common issues section above

---

**Built with**: LangChain • LangGraph • ChromaDB • Groq • HuggingFace
