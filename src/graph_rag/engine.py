from __future__ import annotations

import os
from dataclasses import dataclass

from src.ollama_client import OllamaClient

from .retriever import GraphRetriever, RetrievalResult
from .store import GraphRAGStore


@dataclass
class GraphRAGConfig:
    store_path: str = os.getenv("GRAPHRAG_STORE_PATH", "data/graphrag_store.json")
    chunk_size: int = int(os.getenv("GRAPHRAG_CHUNK_SIZE", "140"))
    chunk_overlap: int = int(os.getenv("GRAPHRAG_CHUNK_OVERLAP", "30"))
    default_top_k: int = int(os.getenv("GRAPHRAG_TOP_K", "4"))
    default_hops: int = int(os.getenv("GRAPHRAG_HOPS", "1"))
    min_relevance_score: float = float(os.getenv("GRAPHRAG_MIN_RELEVANCE_SCORE", "0.18"))


class GraphRAGEngine:
    """Simple GraphRAG: ingest -> graph/vector retrieve -> synthesize answer."""

    def __init__(self, llm: OllamaClient, config: GraphRAGConfig | None = None, store: GraphRAGStore | None = None) -> None:
        self.llm = llm
        self.config = config or GraphRAGConfig()
        self.store = store or GraphRAGStore.load_json(self.config.store_path)
        self.retriever = GraphRetriever(self.store, embed_fn=self._embed)

    def _embed(self, text: str) -> list[float]:
        return self.llm.embed(text)

    def persist(self) -> None:
        self.store.save_json(self.config.store_path)

    def ingest_text(self, text: str, source: str = "manual") -> str:
        doc_id = self.store.add_document(
            text=text,
            source=source,
            embed_fn=self._embed,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
        )
        return doc_id

    def ingest_file(self, file_path: str, source: str | None = None) -> str:
        with open(file_path, "r", encoding="utf-8") as infile:
            text = infile.read()
        chosen_source = source or os.path.basename(file_path)
        return self.ingest_text(text=text, source=chosen_source)

    def retrieve(self, query: str, top_k: int | None = None, hops: int | None = None) -> RetrievalResult:
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