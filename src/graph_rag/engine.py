from __future__ import annotations

import os
from dataclasses import dataclass

from src.ollama_client import OllamaClient

from .retriever import GraphRetriever, RetrievalResult
from .store import GraphRAGStore

@dataclass
class GraphRAGConfig:
    """
    Configuration for GraphRAG operations.
    Attributes:
        store_path (str): File path for the knowledge graph store JSON.
        chunk_size (int): Number of characters per text chunk during ingestion.
        chunk_overlap (int): Number of overlapping characters between chunks.
        default_top_k (int): Default number of top chunks to retrieve.
        default_hops (int): Default number of retrieval hops for multi-hop retrieval.
        min_relevance_score (float): Minimum combined relevance score for retrieval inclusion.
    """    
    
    store_path: str = os.getenv("GRAPHRAG_STORE_PATH", "data/graphrag_store.json")
    chunk_size: int = int(os.getenv("GRAPHRAG_CHUNK_SIZE", "140"))
    chunk_overlap: int = int(os.getenv("GRAPHRAG_CHUNK_OVERLAP", "30"))
    default_top_k: int = int(os.getenv("GRAPHRAG_TOP_K", "4"))
    default_hops: int = int(os.getenv("GRAPHRAG_HOPS", "1"))
    min_relevance_score: float = float(os.getenv("GRAPHRAG_MIN_RELEVANCE_SCORE", "0.18"))

class GraphRAGEngine:
    """
    Orchestrator for GraphRAG operations: ingestion, retrieval, and answer synthesis.

    Manages the knowledge graph store, handles document ingestion with chunking and
    embedding, retrieves relevant context via multi-hop graph traversal, and synthesizes
    answers grounded in the retrieved context.

    Attributes:
        llm (OllamaClient): LLM client for embeddings and text generation.
        config (GraphRAGConfig): Configuration for chunking, retrieval, and synthesis.
        store (GraphRAGStore): The knowledge graph and vector store.
        retriever (GraphRetriever): Performs multi-hop retrieval.

    Example:
        engine = GraphRAGEngine(llm=llm)
        doc_id = engine.ingest_text("Python is a programming language...")
        answer, retrieval = engine.answer("What is Python?")
    """

    def __init__(self, llm: OllamaClient, config: GraphRAGConfig | None = None, store: GraphRAGStore | None = None) -> None:
        self.llm = llm
        self.config = config or GraphRAGConfig()
        self.store = store or GraphRAGStore.load_json(self.config.store_path)
        self.retriever = GraphRetriever(self.store, embed_fn=self._embed)

    def _embed(self, text: str) -> list[float]:
        """
        Generate an embedding for text using the configured LLM.

        Args:
            text (str): Text to embed.

        Returns:
            list[float]: Embedding vector.
        """
        
        return self.llm.embed(text)

    def persist(self) -> None:
        """
        Save the knowledge graph store to disk.

        Writes the store to the configured store_path in JSON format.
        """
        
        self.store.save_json(self.config.store_path)

    def ingest_text(self, text: str, source: str = "manual") -> str:
        """
        Ingest raw text into the knowledge graph.

        Chunks the text, generates embeddings, extracts entities and relationships,
        and adds to the store. Does NOT persist to disk (call persist() explicitly).

        Args:
            text (str): The text to ingest.
            source (str, optional): Source identifier for the document. Defaults to "manual".

        Returns:
            str: Document ID of the ingested text.

        Raises:
            ValueError: If text is empty.
        """
        
        doc_id = self.store.add_document(
            text=text,
            source=source,
            embed_fn=self._embed,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
        )
        return doc_id

    def ingest_file(self, file_path: str, source: str | None = None) -> str:
        """
        Ingest a file into the knowledge graph.

        Reads file content and ingests as text. Does NOT persist to disk.

        Args:
            file_path (str): Path to the file to ingest.
            source (str, optional): Source identifier. Defaults to filename.

        Returns:
            str: Document ID of the ingested file.

        Raises:
            FileNotFoundError: If file does not exist.
        """
        
        with open(file_path, "r", encoding="utf-8") as infile:
            text = infile.read()
        chosen_source = source or os.path.basename(file_path)
        return self.ingest_text(text=text, source=chosen_source)

    def retrieve(self, query: str, top_k: int | None = None, hops: int | None = None) -> RetrievalResult:
        """
        Retrieve relevant context for a query.

        Performs semantic and multi-hop graph retrieval to find relevant chunks.

        Args:
            query (str): The query string.
            top_k (int, optional): Number of top chunks. Defaults to config.default_top_k.
            hops (int, optional): Number of retrieval hops. Defaults to config.default_hops.

        Returns:
            RetrievalResult: Retrieved chunks with scores and context text.
        """
        
        return self.retriever.retrieve(
            query=query,
            top_k=top_k if top_k is not None else self.config.default_top_k,
            hops=hops if hops is not None else self.config.default_hops,
            min_relevance_score=self.config.min_relevance_score,
        )

    def answer(
        self,
        query: str,
        top_k: int | None = None,
        hops: int | None = None,
        temperature: float = 0.2,
    ) -> tuple[str, RetrievalResult]:
        """
        Answer a query using retrieved context and LLM synthesis.

        Retrieves relevant context and generates a grounded answer using the LLM.

        Args:
            query (str): The question to answer.
            top_k (int, optional): Number of top chunks to retrieve.
            hops (int, optional): Number of retrieval hops.
            temperature (float, optional): LLM temperature. Defaults to 0.2.

        Returns:
            tuple[str, RetrievalResult]: The generated answer and retrieval details.
        """
        
        retrieval = self.retrieve(query=query, top_k=top_k, hops=hops)
        prompt = f"""
                Answer the question using the provided knowledge context.
                If the context is insufficient, explicitly say what information is missing.
                When using evidence, cite chunk ids like [doc_1_c0].

                Question:
                {query}

                {retrieval.context_text}
                """.strip()
        answer = self.llm.generate(
            prompt=prompt,
            system="You are a grounded retrieval assistant. Stay faithful to the provided context.",
            temperature=temperature,
        )
        return answer, retrieval