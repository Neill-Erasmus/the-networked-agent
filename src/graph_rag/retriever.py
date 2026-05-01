from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable

from .store import GraphRAGStore

@dataclass
class RetrievalHit:
    """
    Represents a single retrieved chunk.

    Attributes:
        chunk_id (str): Identifier of the chunk (e.g., "doc_1_c0").
        score (float): Relevance score [0-1].
        source (str): Source document identifier.
        text (str): The chunk text content.
        depth (int): Hop distance from initial retrieval.
    """
    
    chunk_id: str
    score: float
    source: str
    text: str
    depth: int
@dataclass
class RetrievalResult:
    """
    Result of a retrieval query.

    Attributes:
        query (str): The original query string.
        hits (list[RetrievalHit]): Retrieved chunks.
        context_text (str): Formatted context for LLM consumption.
    """
    
    query: str
    hits: list[RetrievalHit]
    context_text: str


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """
    Compute cosine similarity between two embedding vectors.

    Args:
        left (list[float]): First embedding vector.
        right (list[float]): Second embedding vector.

    Returns:
        float: Similarity score in range [0.0, 1.0], or 0.0 if vectors are invalid.
    """
    
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


_token_pattern = re.compile(r"[a-zA-Z0-9_]+")


def lexical_similarity(query: str, text: str) -> float:
    """
    Compute lexical similarity as token overlap Jaccard index.

    Ignores tokens with length <= 2.

    Args:
        query (str): Query text.
        text (str): Document text to compare.

    Returns:
        float: Similarity in range [0.0, 1.0].
    """
    
    q_tokens = {token.lower() for token in _token_pattern.findall(query) if len(token) > 2}
    t_tokens = {token.lower() for token in _token_pattern.findall(text) if len(token) > 2}
    if not q_tokens or not t_tokens:
        return 0.0
    return len(q_tokens & t_tokens) / len(q_tokens)
class GraphRetriever:
    """
    Retrieves relevant chunks via vector similarity and multi-hop graph expansion.

    Combines semantic (vector) and lexical similarity scoring, then expands the
    initial retrieval results by following graph edges to neighboring chunks.
    Supports configurable hop distance and neighbor expansion.

    Attributes:
        store (GraphRAGStore): The knowledge graph and vector store.
        embed_fn (Callable): Function to generate embeddings.

    Example:
        retriever = GraphRetriever(store, embed_fn=llm.embed)
        result = retriever.retrieve("What is Python?", top_k=4, hops=1)
    """

    def __init__(self, store: GraphRAGStore, embed_fn: Callable[[str], list[float]]) -> None:
        self.store = store
        self.embed_fn = embed_fn

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        hops: int = 1,
        neighbors_per_hop: int = 2,
        min_relevance_score: float = 0.18,
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks with optional multi-hop graph expansion.

        Process:
        1. Embed query and compute combined vector+lexical scores for all chunks
        2. Select top_k seed chunks above min_relevance_score
        3. Expand via graph neighbors for up to 'hops' iterations
        4. Return all selected chunks, weighted by hop distance

        Args:
            query (str): The query string to retrieve for.
            top_k (int, optional): Number of seed chunks. Defaults to 4.
            hops (int, optional): Number of expansion hops. Defaults to 1.
            neighbors_per_hop (int, optional): Neighbors to explore per hop. Defaults to 2.
            min_relevance_score (float, optional): Score threshold for seed selection. Defaults to 0.18.

        Returns:
            RetrievalResult: Retrieved chunks with context text formatted for LLM.

        Raises:
            ValueError: If query embedding fails or chunks lack embeddings.
        """
        
        top_k = max(1, int(top_k))
        hops = max(0, int(hops))
        neighbors_per_hop = max(1, int(neighbors_per_hop))

        if not self.store.chunks:
            return RetrievalResult(query=query, hits=[], context_text="Knowledge context:\n[No stored knowledge yet]")

        query_embedding = self.embed_fn(query)
        if not query_embedding:
            raise ValueError("Query embedding is empty; GraphRAG requires a working embedding model.")

        base_scores: dict[str, float] = {}
        for chunk_id, chunk in self.store.chunks.items():
            if not chunk.embedding:
                raise ValueError(
                    f"Chunk `{chunk_id}` has no embedding. Re-ingest the knowledge store with a valid embedding model."
                )
            vector_score = cosine_similarity(query_embedding, chunk.embedding)
            lex_score = lexical_similarity(query, chunk.text)
            base_scores[chunk_id] = (0.8 * vector_score) + (0.2 * lex_score)

        ranked_chunk_ids = sorted(base_scores, key=base_scores.get, reverse=True) #type: ignore
        seed_ids = [chunk_id for chunk_id in ranked_chunk_ids if base_scores.get(chunk_id, -1.0) >= min_relevance_score][:top_k]

        selected: dict[str, RetrievalHit] = {}
        frontier = list(seed_ids)
        for chunk_id in seed_ids:
            chunk = self.store.chunks[chunk_id]
            selected[chunk_id] = RetrievalHit(
                chunk_id=chunk_id,
                score=base_scores[chunk_id],
                source=chunk.source,
                text=chunk.text,
                depth=0,
            )

        for hop in range(1, hops + 1):
            if not frontier:
                break
            new_frontier_set: set[str] = set()
            for chunk_id in frontier:
                neighbors = set(self.store.chunk_neighbors.get(chunk_id, set()))
                
                chunk = self.store.chunks.get(chunk_id)
                if chunk:
                    for rel in chunk.relationships:
                        related_chunks = self.store.entity_index.get(rel.target_entity, set())
                        neighbors.update(related_chunks)

                neighbors = list(neighbors)
                neighbors.sort(key=lambda cid: base_scores.get(cid, -1.0), reverse=True)
                
                for neighbor_id in neighbors[:neighbors_per_hop]:
                    chunk = self.store.chunks[neighbor_id]
                    weighted_score = base_scores.get(neighbor_id, 0.0) * (0.9 ** hop)
                    hit = selected.get(neighbor_id)
                    if hit is None or weighted_score > hit.score:
                        selected[neighbor_id] = RetrievalHit(
                            chunk_id=neighbor_id,
                            score=weighted_score,
                            source=chunk.source,
                            text=chunk.text,
                            depth=hop,
                        )
                    new_frontier_set.add(neighbor_id)
            frontier = list(new_frontier_set)

        hits = sorted(selected.values(), key=lambda hit: hit.score, reverse=True)
        max_hits = max(top_k, top_k + hops * neighbors_per_hop)
        hits = hits[:max_hits]

        lines = ["Knowledge context:"]
        if not hits:
            lines.append("[No relevant chunks found]")
        else:
            for hit in hits:
                lines.append(
                    f"[{hit.chunk_id} | source={hit.source} | depth={hit.depth} | score={hit.score:.3f}] {hit.text}"
                )

        return RetrievalResult(query=query, hits=hits, context_text="\n".join(lines))