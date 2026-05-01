from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional


@dataclass
class Relationship:
    """Entity-entity relationship extracted from text."""
    source_entity: str
    target_entity: str
    relation_type: str  # e.g., "produces", "has", "is_type_of"
    chunk_id: str
    confidence: float = 0.5


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    index: int
    source: str
    text: str
    embedding: list[float]
    entities: list[str] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)


@dataclass
class Document:
    doc_id: str
    source: str
    text: str
    chunk_ids: list[str]


class GraphRAGStore:
    """Lightweight knowledge graph + vector store hybrid."""

    def __init__(self) -> None:
        self.documents: dict[str, Document] = {}
        self.chunks: dict[str, Chunk] = {}
        self.chunk_neighbors: dict[str, set[str]] = {}
        self.entity_index: dict[str, set[str]] = {}
        self.relationships: list[Relationship] = []

    @staticmethod
    def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
        words = text.split()
        if not words:
            return []

        overlap = max(0, min(overlap, chunk_size - 1))
        step = max(1, chunk_size - overlap)
        chunks: list[str] = []
        for start in range(0, len(words), step):
            window = words[start : start + chunk_size]
            if not window:
                break
            chunks.append(" ".join(window))
            if start + chunk_size >= len(words):
                break
        return chunks

    @staticmethod
    def _extract_entities(text: str, max_entities: int = 12) -> list[str]:
        proper_nouns = re.findall(r"\b[A-Z][a-zA-Z0-9_-]*(?:\s+[A-Z][a-zA-Z0-9_-]*){0,2}\b", text)
        long_terms = re.findall(r"\b[a-z][a-z0-9_-]{6,}\b", text)

        candidates = proper_nouns + long_terms
        entities: list[str] = []
        seen: set[str] = set()
        for item in candidates:
            normalized = item.strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            entities.append(normalized)
            if len(entities) >= max_entities:
                break
        return entities

    @staticmethod
    def _extract_relationships(text: str, entities: list[str]) -> list[tuple[str, str, str]]:
        """Extract entity-entity relationships from text using pattern matching.
        
        Returns list of (source_entity, target_entity, relation_type) tuples.
        """
        if not entities or len(entities) < 2:
            return []
        
        relationships: list[tuple[str, str, str]] = []
        text_lower = text.lower()
        
        # Define relationship patterns (entity1 -> relation -> entity2)
        relation_patterns = [
            (r"\b{}\b.*?\b(produces|creates|generates)\b.*?\b{}\b", "produces"),
            (r"\b{}\b.*?\b(has|contains|includes)\b.*?\b{}\b", "has"),
            (r"\b{}\b.*?\b(is a|is an|is type of|is kind of)\b.*?\b{}\b", "is_type_of"),
            (r"\b{}\b.*?\b(affects|impacts|influences)\b.*?\b{}\b", "affects"),
            (r"\b{}\b.*?\b(lives in|inhabits|found in)\b.*?\b{}\b", "inhabits"),
            (r"\b{}\b.*?\b(made of|composed of|contains)\b.*?\b{}\b", "made_of"),
            (r"\b{}\b.*?\b(related to|connected to|associated with)\b.*?\b{}\b", "related_to"),
        ]
        
        for i, ent1 in enumerate(entities):
            for j, ent2 in enumerate(entities):
                if i == j or ent1 == ent2:
                    continue
                
                for pattern, rel_type in relation_patterns:
                    try:
                        regex = pattern.format(re.escape(ent1), re.escape(ent2))
                        if re.search(regex, text_lower, re.IGNORECASE):
                            relationships.append((ent1, ent2, rel_type))
                            break
                    except (re.error, IndexError):
                        continue
        
        return relationships

    def _connect(self, left_chunk_id: str, right_chunk_id: str) -> None:
        if left_chunk_id == right_chunk_id:
            return
        self.chunk_neighbors.setdefault(left_chunk_id, set()).add(right_chunk_id)
        self.chunk_neighbors.setdefault(right_chunk_id, set()).add(left_chunk_id)

    def add_document(
        self,
        text: str,
        source: str,
        embed_fn: Callable[[str], list[float]],
        chunk_size: int = 140,
        overlap: int = 30,
    ) -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            raise ValueError("Cannot add an empty document to GraphRAGStore")

        doc_id = f"doc_{len(self.documents) + 1}"
        chunk_texts = self._chunk_text(cleaned, chunk_size=chunk_size, overlap=overlap)
        chunk_ids: list[str] = []

        for idx, chunk_text in enumerate(chunk_texts):
            chunk_id = f"{doc_id}_c{idx}"
            embedding = embed_fn(chunk_text)
            if not embedding:
                raise ValueError(
                    f"Embedding model returned an empty vector for chunk `{chunk_id}`. "
                    "GraphRAG ingestion requires valid embeddings."
                )
            entities = self._extract_entities(chunk_text)
            
            # Extract relationships between entities in this chunk
            rel_tuples = self._extract_relationships(chunk_text, entities)
            relationships = [
                Relationship(
                    source_entity=src,
                    target_entity=tgt,
                    relation_type=rel_type,
                    chunk_id=chunk_id,
                    confidence=0.7,
                )
                for src, tgt, rel_type in rel_tuples
            ]

            chunk = Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                index=idx,
                source=source,
                text=chunk_text,
                embedding=embedding,
                entities=entities,
                relationships=relationships,
            )
            self.chunks[chunk_id] = chunk
            self.chunk_neighbors.setdefault(chunk_id, set())
            chunk_ids.append(chunk_id)

            if idx > 0:
                self._connect(chunk_ids[idx - 1], chunk_id)

            for entity in entities:
                existing = self.entity_index.setdefault(entity, set())
                for prior_chunk_id in list(existing):
                    self._connect(prior_chunk_id, chunk_id)
                existing.add(chunk_id)
            
            # Store extracted relationships for global graph traversal.
            for rel in relationships:
                self.relationships.append(rel)

        self.documents[doc_id] = Document(doc_id=doc_id, source=source, text=cleaned, chunk_ids=chunk_ids)
        return doc_id

    def to_dict(self) -> dict:
        return {
            "documents": {doc_id: asdict(doc) for doc_id, doc in self.documents.items()},
            "chunks": {chunk_id: asdict(chunk) for chunk_id, chunk in self.chunks.items()},
            "chunk_neighbors": {chunk_id: sorted(list(neighbors)) for chunk_id, neighbors in self.chunk_neighbors.items()},
            "entity_index": {entity: sorted(list(chunk_ids)) for entity, chunk_ids in self.entity_index.items()},
            "relationships": [asdict(rel) for rel in self.relationships],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> GraphRAGStore:
        store = cls()
        for doc_id, doc_data in payload.get("documents", {}).items():
            store.documents[doc_id] = Document(**doc_data)
        for chunk_id, chunk_data in payload.get("chunks", {}).items():
            chunk_data_copy = dict(chunk_data)
            relationships_data = chunk_data_copy.pop("relationships", [])
            chunk = Chunk(**chunk_data_copy)
            chunk.relationships = [Relationship(**rel_data) for rel_data in relationships_data]
            store.chunks[chunk_id] = chunk
        for chunk_id, neighbors in payload.get("chunk_neighbors", {}).items():
            store.chunk_neighbors[chunk_id] = set(neighbors)
        for entity, chunk_ids in payload.get("entity_index", {}).items():
            store.entity_index[entity] = set(chunk_ids)
        for rel_data in payload.get("relationships", []):
            rel = Relationship(**rel_data)
            store.relationships.append(rel)
        return store

    def save_json(self, file_path: str) -> None:
        folder = os.path.dirname(file_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as outfile:
            json.dump(self.to_dict(), outfile, indent=2, ensure_ascii=False)

    @classmethod
    def load_json(cls, file_path: str) -> GraphRAGStore:
        if not os.path.exists(file_path):
            return cls()
        with open(file_path, "r", encoding="utf-8") as infile:
            payload = json.load(infile)
        return cls.from_dict(payload)