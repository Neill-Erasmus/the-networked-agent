from __future__ import annotations

import os
from dataclasses import dataclass

from src.graph_of_thought import GoTConfig, GraphOfThoughtController
from src.graph_rag import GraphRAGConfig, GraphRAGEngine
from src.ollama_client import OllamaClient
from src.visualization import QueryVisualizer

def _env_flag(name: str, default: bool) -> bool:
    """
    Convert an environment variable to a boolean flag.

    Args:
        name (str): Environment variable name.
        default (bool): Default value if the environment variable is not set.

    Returns:
        bool: The boolean value of the environment variable or the default.
    """    
    
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}

@dataclass
class NetworkedAgentConfig:
    """
    Configuration for the NetworkedAgent.
    Attributes:
        store_path (str): File path for the GraphRAG store JSON.
        got_max_depth (int): Maximum depth for Graph of Thought reasoning.
        got_beam_width (int): Beam width for Graph of Thought search.
        got_branch_factor (int): Branch factor for Graph of Thought expansions.
        got_temperature (float): Temperature for LLM generation during GoT.
        rag_top_k (int): Number of top chunks to retrieve for initial context.
        rag_hops (int): Number of retrieval hops for multi-hop retrieval.
        save_visualizations (bool): Whether to save visualization artifacts per query.
        visualization_dir (str): Directory to save visualization files.
        enable_dynamic_context (bool): Whether to enable dynamic context refinement during GoT reasoning.
    """    
    
    store_path: str = os.getenv("GRAPHRAG_STORE_PATH", "data/graphrag_store.json")
    got_max_depth: int = int(os.getenv("GOT_MAX_DEPTH", "3"))
    got_beam_width: int = int(os.getenv("GOT_BEAM_WIDTH", "3"))
    got_branch_factor: int = int(os.getenv("GOT_BRANCH_FACTOR", "3"))
    got_temperature: float = float(os.getenv("GOT_TEMPERATURE", "0.2"))
    rag_top_k: int = int(os.getenv("GRAPHRAG_TOP_K", "4"))
    rag_hops: int = int(os.getenv("GRAPHRAG_HOPS", "1"))
    save_visualizations: bool = _env_flag("AGENT_SAVE_VISUALIZATIONS", True)
    visualization_dir: str = os.getenv("AGENT_VISUALIZATION_DIR", "data/visualizations")
    enable_dynamic_context: bool = _env_flag("AGENT_DYNAMIC_CONTEXT", True)

@dataclass
class EpisodicMemory:
    """
    Represents a single turn of agent reasoning and response, stored in episodic memory.
    This is separate from the knowledge base to prevent contamination with synthetic outputs.
    Attributes:
        query (str): The user's question or task.
        answer (str): The agent's response to the query.
        reasoning_path (list[str]): The sequence of reasoning steps taken to arrive at the answer.
        retrieved_chunk_ids (list[str]): IDs of knowledge chunks retrieved and used in reasoning.
        confidence (float): Optional confidence score for the answer (0.0 to 1.0).
    """    
    
    query: str
    answer: str
    reasoning_path: list[str]
    retrieved_chunk_ids: list[str]
    confidence: float = 0.5

@dataclass
class AgentTurn:
    """Represents a single turn of agent reasoning and response.
    Attributes:
        question (str): The user's question or task.
        answer (str): The agent's response to the question.
        reasoning_path (list[str]): The sequence of reasoning steps taken to arrive at the answer.
        retrieved_chunk_ids (list[str]): IDs of knowledge chunks retrieved and used in reasoning.
        visualization_html (str): Optional file path to an HTML visualization of the reasoning process.
    """
    
    question: str
    answer: str
    reasoning_path: list[str]
    retrieved_chunk_ids: list[str]
    visualization_html: str = ""

class NetworkedAgent:
    """
    A networked agent that integrates Graph of Thought reasoning with GraphRAG retrieval.

    Raises:
        FileNotFoundError: If an ingested file is not found.
        ValueError: If an unsupported file type is provided.

    Returns:
        _type_: AgentTurn containing the question, answer, reasoning path, retrieved chunk IDs, and optional visualization path.
    """    

    ALLOWED_INGEST_EXTENSIONS: tuple[str, ...] = (".txt", ".json")
    INSUFFICIENT_INFO_MESSAGE: str = (
        "I do not have sufficient information in the current GraphRAG store to assist with this request. "
        "Please ingest relevant text first, then ask your question again."
    )

    def __init__(self, llm: OllamaClient, config: NetworkedAgentConfig | None = None) -> None:
        """
        Initialize the NetworkedAgent with the given LLM client and configuration.

        Args:
            llm (OllamaClient): An instance of the OllamaClient for LLM interactions.
            config (NetworkedAgentConfig | None, optional): The configuration for the networked agent. Defaults to None.
        """        
        
        self.llm = llm
        self.config = config or NetworkedAgentConfig()
        self.visualizer = QueryVisualizer(base_dir=self.config.visualization_dir)
        self.episodic_memory: list[EpisodicMemory] = []
        self.got_controller = GraphOfThoughtController(
            llm=self.llm,
            config=GoTConfig(
                max_depth=self.config.got_max_depth,
                beam_width=self.config.got_beam_width,
                branch_factor=self.config.got_branch_factor,
                temperature=self.config.got_temperature,
            ),
        )
        self.rag_engine = GraphRAGEngine(
            llm=self.llm,
            config=GraphRAGConfig(
                store_path=self.config.store_path,
                default_top_k=self.config.rag_top_k,
                default_hops=self.config.rag_hops,
            ),
        )

    def ingest_text(self, text: str, source: str = "manual") -> str:
        """
        Ingest raw text into the GraphRAG store as a new document.

        Args:
            text (str): The raw text to ingest.
            source (str, optional): The source of the text. Defaults to "manual".

        Returns:
            str: The ID of the ingested document.
        """        
        
        doc_id = self.rag_engine.ingest_text(text=text, source=source)
        self.rag_engine.persist()
        return doc_id

    def ingest_file(self, file_path: str) -> str:
        """
        Ingest a file's content into the GraphRAG store as a new document.

        Args:
            file_path (str): The path to the file to ingest.

        Raises:
            FileNotFoundError: If the file is not found.
            ValueError: If the file type is not supported.

        Returns:
            str: The ID of the ingested document.
        """               
        
        normalized_path = os.path.abspath(file_path)
        if not os.path.isfile(normalized_path):
            raise FileNotFoundError(f"File not found: {normalized_path}")

        extension = os.path.splitext(normalized_path)[1].lower()
        if extension not in self.ALLOWED_INGEST_EXTENSIONS:
            allowed = ", ".join(self.ALLOWED_INGEST_EXTENSIONS)
            raise ValueError(
                f"Unsupported file type `{extension or '[no extension]'}`. "
                f"Supported file types: {allowed}."
            )

        doc_id = self.rag_engine.ingest_file(normalized_path)
        self.rag_engine.persist()
        return doc_id

    def _make_retrieval_callback(self):
        """
        Create a retrieval callback function for dynamic context refinement during GoT reasoning.
        """
        
        def refine_context(task: str, _current_context: str, depth: int) -> str:
            """
            Refine the retrieval context based on the current reasoning step.
            """
            if not self.config.enable_dynamic_context:
                return _current_context

            adaptive_top_k = self.config.rag_top_k + (depth - 1)
            adaptive_hops = min(self.config.rag_hops + (depth - 1) // 2, 3)

            new_retrieval = self.rag_engine.retrieve(
                query=task,
                top_k=adaptive_top_k,
                hops=adaptive_hops,
            )
            return new_retrieval.context_text
        
        return refine_context

    def think_and_answer(self, question: str) -> AgentTurn:
        """
        Reason and answer using GoT with dynamic context refinement.

        Args:
            question (str): The question to answer.

        Returns:
            AgentTurn: The agent's response.
        """       
         
        retrieval = self.rag_engine.retrieve(
            query=question,
            top_k=self.config.rag_top_k,
            hops=self.config.rag_hops,
        )

        if not retrieval.hits:
            fallback_answer = self.INSUFFICIENT_INFO_MESSAGE
            episodic = EpisodicMemory(
                query=question,
                answer=fallback_answer,
                reasoning_path=[],
                retrieved_chunk_ids=[],
                confidence=0.0,
            )
            self.episodic_memory.append(episodic)
            return AgentTurn(
                question=question,
                answer=fallback_answer,
                reasoning_path=[],
                retrieved_chunk_ids=[],
                visualization_html="",
            )
        
        retrieval_callback = self._make_retrieval_callback()
        
        got_result = self.got_controller.solve(
            task=question,
            context=retrieval.context_text,
            retrieval_fn=retrieval_callback if self.config.enable_dynamic_context else None,
        )

        visualization_html = ""
        if self.config.save_visualizations:
            viz_result = self.visualizer.save_turn(
                query=question,
                answer=got_result.answer,
                got_snapshot=got_result.graph_snapshot,
                reasoning_path=got_result.reasoning_path,
                retrieval=retrieval,
                rag_store=self.rag_engine.store,
                chat_model=self.llm.config.chat_model,
                embedding_model=self.llm.config.embedding_model,
            )
            visualization_html = viz_result.html_path

        episodic = EpisodicMemory(
            query=question,
            answer=got_result.answer,
            reasoning_path=got_result.reasoning_path,
            retrieved_chunk_ids=[hit.chunk_id for hit in retrieval.hits],
            confidence=0.7 if got_result.best_thought else 0.3,
        )
        self.episodic_memory.append(episodic)

        retrieved_chunk_ids = [hit.chunk_id for hit in retrieval.hits]

        return AgentTurn(
            question=question,
            answer=got_result.answer,
            reasoning_path=got_result.reasoning_path,
            retrieved_chunk_ids=retrieved_chunk_ids,
            visualization_html=visualization_html,
        )

    def run_interactive(self) -> None:
        """
        Run an interactive command-line session with the agent.
            Commands:
                /exit - Exit the interactive session.
                /ask <question> - Ask a question to the agent.
                /ingest <file_path> - Ingest a file into the knowledge graph (supported types: .txt, .json).
        """        
        
        print("Networked agent ready.")
        print("Commands:")
        print("  /exit")
        print("  /ask <question>")
        print("  /ingest <file_path>")
        print("    Supported file types: .txt, .json")

        while True:
            user_input = input("\nYou> ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"/exit", "exit", "quit"}:
                print("Goodbye.")
                break

            if user_input.startswith("/ingest "):
                file_path = user_input[len("/ingest ") :].strip()
                try:
                    doc_id = self.ingest_file(file_path)
                    print(f"\nIngested file as document: {doc_id}")
                except Exception as exc:
                    print(f"\nIngestion failed: {exc}")
                continue

            if user_input.startswith("/ask "):
                question = user_input[len("/ask ") :].strip()
            else:
                question = user_input

            result = self.think_and_answer(question)
            print("\nAgent>")
            print(result.answer)
            if result.retrieved_chunk_ids:
                print("\nRetrieved chunks: " + ", ".join(result.retrieved_chunk_ids))
            if result.reasoning_path:
                print("\nReasoning path:")
                for idx, step in enumerate(result.reasoning_path, start=1):
                    print(f"{idx}. {step}")
            if result.visualization_html:
                print("\nVisualization file:")
                print(result.visualization_html)