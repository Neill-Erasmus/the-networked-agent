# The Networked Agent

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-FF6B6B?style=flat-square)](https://ollama.ai)

A sophisticated AI reasoning framework that combines **Graph of Thought (GoT)** decomposition with **GraphRAG** retrieval to enable powerful multi-step problem-solving with local LLMs.

---

## Summary

### What Problem Does This Solve?

Most LLM applications struggle with complex reasoning tasks because they attempt to solve everything in a single forward pass. The Networked Agent addresses this by separating **knowledge retrieval** from **reasoning**, allowing an AI system to:

1. **Break down complex problems** into manageable subproblems
2. **Retrieve relevant context** from a knowledge base at each reasoning step
3. **Explore multiple reasoning paths** and score them for quality
4. **Synthesize final answers** with full transparency into the reasoning process

### Graph of Thought (GoT): Structured Reasoning

**Graph of Thought** is a reasoning framework that treats problem-solving as graph exploration rather than linear token generation:

- **Decomposition**: Complex tasks are broken into a hierarchy of subproblems
- **Expansion**: Each thought node is expanded into multiple candidate branches
- **Scoring**: Candidate thoughts are evaluated for quality and relevance
- **Beam Search**: Only the most promising paths are explored (configurable beam width)
- **Synthesis**: The best reasoning path is assembled into a final answer

**Real-world analogy**: Instead of a student writing an essay in one draft, GoT is like a student outlining topics, exploring multiple explanations for each point, evaluating which explanations are strongest, and then synthesizing the best path into a coherent essay.

### GraphRAG: Knowledge Graph Retrieval-Augmented Generation

**GraphRAG** is a retrieval system that goes beyond simple document matching:

- **Knowledge Graph Construction**: Ingested text is parsed into entities and relationships
- **Multi-hop Retrieval**: Context is retrieved not just from direct matches, but from neighboring nodes in the knowledge graph (configurable hop distance)
- **Vector Similarity**: Semantic search finds relevant chunks even with paraphrasing
- **Evidence Citation**: Retrieved chunks are tracked and can be cited in final answers

**Real-world analogy**: Instead of a library search that finds one matching book, GraphRAG is like following citation chains—starting with relevant books and then checking what *those* books reference to build a richer context.

### The Networked Agent: Integration Architecture

The Networked Agent combines these two systems:

- **GoT** breaks down the user's question into subproblems
- **GraphRAG** retrieves contextual knowledge for each subproblem
- **GoT** explores multiple reasoning paths with this dynamic context
- **Episodic Memory** tracks the reasoning journey separately from the source knowledge base
- **Visualizations** provide full transparency into what the agent found and how it reasoned

This is particularly powerful for:
- Complex multi-step reasoning over domain knowledge
- Research synthesis combining multiple sources
- Debugging-style tasks requiring systematic exploration
- Interactive learning with dynamic context windows

---

## Key Features

✨ **Advanced Reasoning**
- Decomposition-based problem-solving with configurable depth and branching
- Beam search exploration of promising reasoning paths
- Scoring and ranking of candidate thoughts
- Episodic memory to track agent reasoning history

🔗 **Knowledge Integration**
- Multi-hop graph-based retrieval over ingested documents
- Automatic knowledge graph construction from text
- Vector similarity search with semantic understanding
- Evidence citation and source tracking

📊 **Transparency & Debugging**
- Full reasoning path visualization
- HTML outputs showing GoT and GraphRAG artifacts
- Graph representations (DOT format) for both systems
- Configurable logging and inspection

🏠 **Fully Local**
- Powered by Ollama for private, on-device LLM inference
- No API calls or external dependencies
- Full control over models and parameters

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    NetworkedAgent                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  User Query / Task                                   │  │
│  └──────────────────────────────────────────────────────┘  │
│                          │                                  │
│                          ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Graph of Thought Controller                         │  │
│  │  ├─ Decompose task into subproblems                  │  │
│  │  ├─ Expand each thought node with branching          │  │
│  │  ├─ Score candidate thoughts                         │  │
│  │  └─ Beam search over promising paths                 │  │
│  └──────────────────────────────────────────────────────┘  │
│                          │                                  │
│          ┌───────────────┴───────────────┐                 │
│          │ For each reasoning step...    │                 │
│          ▼                               │                 │
│  ┌──────────────────────────────────┐   │                 │
│  │ GraphRAG Engine                  │   │                 │
│  │ ├─ Retrieve relevant context     │   │                 │
│  │ ├─ Multi-hop graph traversal     │   │                 │
│  │ └─ Return enhanced context       │   │                 │
│  └──────────────────────────────────┘   │                 │
│          │                               │                 │
│          └───────────────┬───────────────┘                 │
│                          ▼                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Final Synthesis & Episodic Memory                   │  │
│  │  ├─ Best reasoning path selected                     │  │
│  │  ├─ Answer synthesized from path                     │  │
│  │  └─ Conversation history recorded                    │  │
│  └──────────────────────────────────────────────────────┘  │
│                          │                                  │
│                          ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Visualization Output (HTML + JSON + DOT)            │  │
│  │  ├─ Reasoning graph                                  │  │
│  │  ├─ Retrieval chains                                 │  │
│  │  └─ Evidence citations                               │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

- **Python 3.10+**
- **Ollama** (running locally or remotely)
- Ollama models installed:
  - Chat model: `ollama pull llama3` (or your preferred model)
  - Embedding model: `ollama pull nomic-embed-text`

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd the-networked-agent
   ```

2. **Create a virtual environment**
   ```bash
   python3.10 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start Ollama**
   ```bash
   ollama serve
   ```

5. **Ensure models are available**
   ```bash
   ollama pull llama3
   ollama pull nomic-embed-text
   ```

---

## Quick Start

### Basic Usage

```python
from networked_agent import NetworkedAgent
from src.ollama_client import OllamaClient, OllamaConfig

# Initialize LLM client
llm = OllamaClient(
    OllamaConfig(
        base_url="http://localhost:11434",
        chat_model="llama3:latest",
        embedding_model="nomic-embed-text",
    )
)

# Create agent
agent = NetworkedAgent(llm=llm)

# Ingest knowledge
doc_id = agent.ingest_text("Python is a high-level programming language...")
agent.save()

# Ask a question
answer = agent.query("What are the benefits of Python?")
print(answer.answer)
print(f"Confidence: {answer.confidence}")
```

### Command Line

**Graph of Thought (standalone)**
```bash
python -m src.graph_of_thought.main \
  --task "How do I implement a binary search?" \
  --context "You can use a sorted array" \
  --max-depth 3 \
  --beam-width 3
```

**GraphRAG (standalone)**
```bash
python -m src.graph_rag.main \
  --query "What is mentioned about Python?" \
  --store-path data/graphrag_store.json \
  --top-k 4 \
  --hops 2
```

---

## Configuration

### Environment Variables

#### Graph of Thought
- `GOT_MAX_DEPTH`: Maximum reasoning depth (default: 3)
- `GOT_BEAM_WIDTH`: Beam width for search (default: 3)
- `GOT_BRANCH_FACTOR`: Branches per node expansion (default: 3)
- `GOT_TEMPERATURE`: LLM temperature for reasoning (default: 0.2)

#### GraphRAG
- `GRAPHRAG_STORE_PATH`: Path to knowledge graph store (default: `data/graphrag_store.json`)
- `GRAPHRAG_TOP_K`: Number of top chunks to retrieve (default: 4)
- `GRAPHRAG_HOPS`: Multi-hop retrieval distance (default: 1)
- `GRAPHRAG_CHUNK_SIZE`: Document chunk size (default: 140)
- `GRAPHRAG_CHUNK_OVERLAP`: Chunk overlap for sliding window (default: 30)
- `GRAPHRAG_MIN_RELEVANCE_SCORE`: Minimum similarity threshold (default: 0.18)

#### Ollama
- `OLLAMA_BASE_URL`: Ollama server URL (default: `http://localhost:11434`)
- `OLLAMA_CHAT_MODEL`: Chat model name (default: `llama3:latest`)
- `OLLAMA_EMBED_MODEL`: Embedding model name (default: `nomic-embed-text`)

#### Agent
- `AGENT_SAVE_VISUALIZATIONS`: Save HTML visualizations (default: `true`)
- `AGENT_VISUALIZATION_DIR`: Output directory for visualizations (default: `data/visualizations`)
- `AGENT_DYNAMIC_CONTEXT`: Dynamically retrieve context during reasoning (default: `true`)

### Configuration Objects

```python
from networked_agent import NetworkedAgentConfig
from src.graph_of_thought import GoTConfig
from src.graph_rag import GraphRAGConfig

got_config = GoTConfig(
    max_depth=4,
    beam_width=5,
    branch_factor=3,
    temperature=0.3,
)

rag_config = GraphRAGConfig(
    store_path="data/my_store.json",
    default_top_k=6,
    default_hops=2,
    chunk_size=200,
)

agent_config = NetworkedAgentConfig(
    got_max_depth=4,
    rag_top_k=6,
    save_visualizations=True,
)
```

---

## Project Structure

```
the-networked-agent/
├── networked_agent.py           # Main agent orchestrator
├── ollama_client.py             # LLM client wrapper
├── visualization.py             # HTML visualization engine
├── src/
│   ├── graph_of_thought/
│   │   ├── controller.py        # GoT orchestration
│   │   ├── graph.py             # Thought graph data structure
│   │   ├── parser.py            # LLM output parsing
│   │   ├── prompter.py          # Prompt generation
│   │   ├── scorer.py            # Thought evaluation
│   │   ├── search.py            # Beam search algorithm
│   │   └── main.py              # CLI entry point
│   └── graph_rag/
│       ├── engine.py            # RAG orchestration
│       ├── retriever.py         # Multi-hop retrieval
│       ├── store.py             # Knowledge graph storage
│       └── main.py              # CLI entry point
├── data/
│   ├── graphrag_store.json      # Knowledge graph (generated)
│   └── visualizations/          # Output HTML/JSON (generated)
└── README.md                    # This file
```

---

## How to Use: Detailed Workflows

### Workflow 1: Simple RAG Query

```python
from networked_agent import NetworkedAgent
from src.ollama_client import OllamaClient, OllamaConfig

llm = OllamaClient(OllamaConfig())
agent = NetworkedAgent(llm=llm)

# Ingest documents
agent.ingest_text("The Earth orbits the Sun in approximately 365 days.")
agent.ingest_file("path/to/document.txt")
agent.save()

# Query with retrieval
turn = agent.query("How long does Earth take to orbit the Sun?")
print(turn.answer)
```

### Workflow 2: Multi-Step Reasoning with GoT

```python
# Agent automatically uses GoT for complex queries
turn = agent.query(
    "Explain the relationship between Earth's orbit and seasons, "
    "considering axial tilt and solar radiation."
)

# Access detailed reasoning
print(f"Reasoning path: {turn.reasoning_path}")
print(f"Retrieved chunks: {turn.retrieved_chunk_ids}")
print(f"Visualization: {turn.visualization_html}")
```

### Workflow 3: Standalone Graph of Thought

```python
from src.graph_of_thought import GraphOfThoughtController, GoTConfig
from src.ollama_client import OllamaClient, OllamaConfig

llm = OllamaClient(OllamaConfig())
controller = GraphOfThoughtController(
    llm=llm,
    config=GoTConfig(max_depth=3, beam_width=3)
)

# Reason without retrieval (useful for logical puzzles, math, etc.)
result = controller.reason(
    task="If A implies B, and B implies C, what can we conclude about A and C?",
    context=""
)

print(result.answer)
print(result.reasoning_path)
```

### Workflow 4: Visualization & Analysis

```python
turn = agent.query("Complex question here...")

# Inspect results
print(f"Confidence score: {turn.confidence}")
print(f"HTML visualization: {turn.visualization_html}")

# Load visualization in browser
import webbrowser
webbrowser.open(f"file://{turn.visualization_html}")
```

---

## Understanding the Output

### Episodic Memory

The agent maintains separate episodic memory from source knowledge:
- **Source Knowledge**: The ingested documents in GraphRAG store
- **Episodic Memory**: Records of past queries, reasoning paths, and answers
- This separation prevents reasoning artifacts from contaminating factual knowledge

### Confidence Scores

Each query result includes a confidence score (0.0-1.0) based on:
- Quality of retrieved context
- Consistency of reasoning path
- Agreement with source evidence

### Visualization Files

For each query, if `save_visualizations=true`:
- `query_{timestamp}.html`: Interactive visualization
- `query_{timestamp}_got.json`: GoT graph structure
- `query_{timestamp}_rag.json`: RAG retrieval chains
- `query_{timestamp}_got.dot`: GraphViz format for GoT
- `query_{timestamp}_rag.dot`: GraphViz format for RAG

---

## Advanced Topics

### Custom Retrieval Integration

```python
def custom_retriever(task: str, context: str, depth: int) -> str:
    """Custom retrieval function called at each GoT depth."""
    # Your custom retrieval logic here
    return new_context

# Use with GoT
controller.reason(task="...", context="...", retrieval_fn=custom_retriever)
```

### Modifying Prompts

```python
from src.graph_of_thought import GoTPrompter

prompter = GoTPrompter()

# Customize prompts for your domain
custom_system = "You are an expert in software architecture."
custom_decompose = "Break down this software design question: {task}"

# Override and use
result = controller.reason(task="...", context="...")
```

### Scaling to Larger Knowledge Bases

For production use with large document collections:

1. **Store splitting**: Use MongoDB or other backends instead of JSON files
2. **Vector database**: Replace in-memory vectors with Pinecone, Weaviate, etc.
3. **Caching**: Cache LLM responses and retrievals
4. **Batching**: Process multiple queries in parallel

---

## Troubleshooting

### "Ollama server is not reachable"
- Ensure Ollama is running: `ollama serve`
- Check the base URL: default is `http://localhost:11434`
- Verify firewall rules if using remote Ollama

### "Model not found"
```bash
ollama list                    # See available models
ollama pull llama3            # Download model
ollama pull nomic-embed-text  # Download embeddings
```

### Poor retrieval results
- Increase `GRAPHRAG_TOP_K` to retrieve more context
- Increase `GRAPHRAG_HOPS` for multi-hop retrieval
- Lower `GRAPHRAG_MIN_RELEVANCE_SCORE` to include borderline matches
- Ingest more relevant documents

### Slow reasoning
- Reduce `GOT_MAX_DEPTH` to search more shallowly
- Reduce `GOT_BEAM_WIDTH` to explore fewer branches
- Use a faster model (e.g., `mistral` instead of `llama3`)

---

## Performance Characteristics

| Component | Typical Time | Factors |
|-----------|-------------|---------|
| Simple RAG query | 5-15s | Model speed, chunk count |
| GoT reasoning (depth=3) | 30-90s | LLM, branching, retrieval |
| Graph construction | 1-5s per doc | Document size, embedding model |
| Multi-hop retrieval | 2-5x retrieval time | Hop count, graph density |

---

## License

MIT License - See LICENSE file for details

---

## Citation

If you use The Networked Agent in research, please cite:

```bibtex
@software{networked_agent_2026,
  title={The Networked Agent: Graph of Thought + GraphRAG Integration},
  author={Neill Jean Erasmus},
  year={2026},
  url={https://github.com/Neill-Erasmus/the-networked-agent}
}
```