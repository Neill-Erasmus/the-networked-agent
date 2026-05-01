# The Networked Agent

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-FF6B6B?style=flat-square)](https://ollama.ai)

A local AI reasoning framework that combines **Graph of Thought (GoT)** decomposition with **GraphRAG** retrieval to support multi-step problem solving with Ollama.

## Table of Contents

- [Summary](#summary)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [How to Use: Detailed Workflows](#how-to-use-detailed-workflows)
- [Understanding the Output](#understanding-the-output)
- [Advanced Topics](#advanced-topics)
- [Troubleshooting](#troubleshooting)
- [Performance Characteristics](#performance-characteristics)
- [License](#license)
- [Citation](#citation)

---

## Summary

### What Problem Does This Solve?

Most LLM applications struggle with complex reasoning tasks because they attempt to solve everything in a single forward pass. The Networked Agent separates **knowledge retrieval** from **reasoning**, allowing the system to:

1. **Break down complex problems** into manageable subproblems
2. **Retrieve relevant context** from a knowledge base before and during reasoning
3. **Explore multiple reasoning paths** and score them for quality
4. **Synthesize final answers** while preserving the reasoning trace

### Graph of Thought (GoT): Structured Reasoning

**Graph of Thought** treats problem solving as graph exploration rather than linear token generation:

- **Decomposition**: Tasks are split into subproblems with JSON-based prompts
- **Expansion**: Each thought node is expanded into multiple candidate branches
- **Scoring**: Candidate thoughts are evaluated with a hybrid heuristic and LLM score
- **Beam Search**: Only the most promising paths are explored
- **Synthesis**: The best reasoning path is assembled into a final answer

The implementation in this repository lives in [src/graph_of_thought/controller.py](src/graph_of_thought/controller.py), [src/graph_of_thought/search.py](src/graph_of_thought/search.py), and related helpers.

### GraphRAG: Knowledge Graph Retrieval-Augmented Generation

**GraphRAG** goes beyond simple document matching:

- **Knowledge Graph Construction**: Ingested text is chunked, embedded, and indexed with heuristic entity and relationship extraction
- **Multi-hop Retrieval**: Context is retrieved not just from direct matches, but from neighboring chunks in the knowledge graph
- **Vector Similarity**: Semantic search finds relevant chunks even with paraphrasing
- **Evidence Tracking**: Retrieved chunk ids, source labels, and hop depth are preserved in the retrieval result

The implementation in this repository lives in [src/graph_rag/engine.py](src/graph_rag/engine.py), [src/graph_rag/retriever.py](src/graph_rag/retriever.py), and [src/graph_rag/store.py](src/graph_rag/store.py).

### The Networked Agent: Integration Architecture

The Networked Agent combines these two systems:

- **GoT** breaks down the user's question into subproblems
- **GraphRAG** retrieves contextual knowledge before reasoning and can refine context during search
- **Episodic Memory** tracks the reasoning journey separately from the source knowledge base
- **Visualizations** persist HTML, JSON, and DOT artifacts for each answer turn

This is particularly powerful for:
- Complex multi-step reasoning over domain knowledge
- Research synthesis combining multiple sources
- Debugging-style tasks requiring systematic exploration
- Interactive learning with dynamic context windows

---

## Key Features

✨ **Advanced Reasoning**
- Decomposition-based problem solving with configurable depth, beam width, branch factor, and temperature
- Beam search exploration of promising reasoning paths
- Hybrid scoring that combines heuristics with optional LLM-based scoring
- Episodic memory to track agent reasoning history

🔗 **Knowledge Integration**
- Multi-hop graph-based retrieval over ingested documents
- Automatic document chunking and embedding on ingestion
- Vector similarity search with lexical overlap as a secondary signal
- Evidence tracking with chunk ids, sources, and hop depth

📊 **Transparency & Debugging**
- Full reasoning path visualization
- HTML outputs showing GoT and GraphRAG artifacts
- Graph representations in JSON and DOT format for both systems
- Configurable output directory and optional visualization disabling

🏠 **Fully Local**
- Powered by Ollama for private, on-device LLM inference
- No hosted API calls are required
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
- **Ollama** running locally or remotely
- Ollama models installed:
  - Chat model: `ollama pull llama3` or your preferred chat model
  - Embedding model: `ollama pull nomic-embed-text` or your preferred embedding model
- No Python package install step is required; this repository currently uses the Python standard library plus Ollama's HTTP API

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

3. **Start Ollama**
   ```bash
   ollama serve
   ```

4. **Ensure models are available**
   ```bash
   ollama pull llama3
   ollama pull nomic-embed-text
   ```

---

## Quick Start

### Basic Usage

```python
from src.networked_agent import NetworkedAgent
from src.ollama_client import OllamaClient, OllamaConfig

llm = OllamaClient(
    OllamaConfig(
        base_url="http://localhost:11434",
        chat_model="llama3:latest",
        embedding_model="nomic-embed-text",
    )
)

agent = NetworkedAgent(llm=llm)

doc_id = agent.ingest_text("Python is a high-level programming language...")

turn = agent.think_and_answer("What are the benefits of Python?")
print(turn.answer)
print(turn.retrieved_chunk_ids)
print(turn.reasoning_path)
print(turn.visualization_html)
```

If you want the latest turn stored in memory, inspect `agent.episodic_memory[-1]` after a query.

### Command Line

**Root CLI**
```bash
python main.py --ask "What are the benefits of Python?"
```

**Ingest a text or JSON file**
```bash
python main.py --ingest-file path/to/document.txt
```

**Interactive mode**
```bash
python main.py --interactive
```

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
- `GRAPHRAG_CHUNK_SIZE`: Word count per chunk during ingestion (default: 140)
- `GRAPHRAG_CHUNK_OVERLAP`: Word overlap between neighboring chunks (default: 30)
- `GRAPHRAG_MIN_RELEVANCE_SCORE`: Minimum retrieval threshold for seed chunks (default: 0.18)

#### Ollama
- `OLLAMA_BASE_URL`: Ollama server URL (default: `http://localhost:11434`)
- `OLLAMA_CHAT_MODEL`: Chat model name (default: `llama3:latest`)
- `OLLAMA_EMBED_MODEL`: Embedding model name (default: `nomic-embed-text:latest`)
- `OLLAMA_TIMEOUT_SECONDS`: Request timeout for Ollama HTTP calls in seconds (default: 600)

#### Agent
- `AGENT_SAVE_VISUALIZATIONS`: Save HTML visualizations (default: `true`)
- `AGENT_VISUALIZATION_DIR`: Output directory for visualizations (default: `data/visualizations`)
- `AGENT_DYNAMIC_CONTEXT`: Dynamically retrieve context during reasoning (default: `true`)

### Configuration Objects

```python
from src.networked_agent import NetworkedAgentConfig
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
    chunk_overlap=40,
)

agent_config = NetworkedAgentConfig(
    got_max_depth=4,
    rag_top_k=6,
    save_visualizations=True,
)
```

---

## How to Use: Detailed Workflows

### Workflow 1: Simple RAG Query

```python
from src.networked_agent import NetworkedAgent
from src.ollama_client import OllamaClient, OllamaConfig

llm = OllamaClient(OllamaConfig())
agent = NetworkedAgent(llm=llm)

agent.ingest_text("The Earth orbits the Sun in approximately 365 days.")
agent.ingest_file("path/to/document.txt")

turn = agent.think_and_answer("How long does Earth take to orbit the Sun?")
print(turn.answer)
print(turn.retrieved_chunk_ids)
```

### Workflow 2: Multi-Step Reasoning with GoT

```python
turn = agent.think_and_answer(
    "Explain the relationship between Earth's orbit and seasons, "
    "considering axial tilt and solar radiation."
)

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

result = controller.solve(
    task="If A implies B, and B implies C, what can we conclude about A and C?",
    context=""
)

print(result.answer)
print(result.best_thought)
print(result.reasoning_path)
```

### Workflow 4: Visualization & Analysis

```python
turn = agent.think_and_answer("Complex question here...")

print(f"HTML visualization: {turn.visualization_html}")

import webbrowser
webbrowser.open(turn.visualization_html)
```

---

## Understanding the Output

### Episodic Memory

The agent maintains separate episodic memory from source knowledge:
- **Source Knowledge**: The ingested documents in the GraphRAG store
- **Episodic Memory**: Records of past queries, reasoning paths, and answers
- This separation prevents reasoning artifacts from contaminating factual knowledge

### Confidence Scores

Confidence is not exposed on the public `AgentTurn` object. The agent does store a lightweight internal confidence value in episodic memory, but the public outputs are the answer, reasoning path, retrieved chunk ids, and optional visualization path.

### Visualization Files

For each query, if `AGENT_SAVE_VISUALIZATIONS=true`:
- `visualization.html`: Interactive visualization
- `got_graph.json`: GoT graph structure
- `graphrag_graph.json`: GraphRAG retrieval graph and hits
- `got_graph.dot`: GraphViz format for GoT
- `graphrag_graph.dot`: GraphViz format for GraphRAG
- `meta.json`: Query metadata, including models and output directory
- Files are written into a timestamped directory under `AGENT_VISUALIZATION_DIR`

---

## Advanced Topics

### Custom Retrieval Integration

```python
def custom_retriever(task: str, context: str, depth: int) -> str:
    """Custom retrieval function called at each GoT depth."""
    return new_context

result = controller.solve(task="...", context="...", retrieval_fn=custom_retriever)
```

### Modifying Prompts

```python
from src.graph_of_thought import GoTPrompter

prompter = GoTPrompter()

custom_system = "You are an expert in software architecture."
custom_decompose = "Break down this software design question: {task}"

result = controller.solve(task="...", context="...")
```

### Scaling to Larger Knowledge Bases

For production use with large document collections:

1. **Store splitting**: Move the JSON-backed store to a database-backed implementation
2. **Vector database**: Replace in-memory embeddings with a dedicated vector store
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
ollama pull llama3             # Download model
ollama pull nomic-embed-text   # Download embeddings
```

### "Model name is empty"
- Set `OLLAMA_CHAT_MODEL` and `OLLAMA_EMBED_MODEL` explicitly if your environment does not already provide them
- The code validates that both models exist before starting a query

### Poor retrieval results
- Increase `GRAPHRAG_TOP_K` to retrieve more context
- Increase `GRAPHRAG_HOPS` for multi-hop retrieval
- Lower `GRAPHRAG_MIN_RELEVANCE_SCORE` to include borderline matches
- Ingest more relevant documents
- Check that the embedding model is returning non-empty vectors

### Slow reasoning
- Reduce `GOT_MAX_DEPTH` to search more shallowly
- Reduce `GOT_BEAM_WIDTH` to explore fewer branches
- Reduce `GOT_BRANCH_FACTOR` to generate fewer expansions
- Use a faster model if your local Ollama setup supports one

---

## Performance Characteristics

| Component | Typical Time | Factors |
|-----------|-------------|---------|
| Simple RAG query | Depends on model latency | Model speed, chunk count, retrieval depth |
| GoT reasoning (depth=3) | Depends on model latency | LLM speed, beam width, branch factor, retrieval refinement |
| Graph construction | Depends on document size | Chunk count, embedding model, relationship extraction |
| Multi-hop retrieval | Usually slower than single-hop retrieval | Hop count, graph density, selected seed chunks |

---

## License

MIT License - See [LICENSE](License) file for details