from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ThoughtNode:
	node_id: str
	content: str
	depth: int
	parent_id: Optional[str]
	score: float = 0.0
	rationale: str = ""
	metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ThoughtEdge:
	source_id: str
	target_id: str
	relation: str = "expands"


class ThoughtGraph:
	"""In-memory graph containing all generated thoughts and relations."""

	def __init__(self) -> None:
		self.nodes: dict[str, ThoughtNode] = {}
		self.edges: list[ThoughtEdge] = []
		self._counter = 0

	def _next_id(self) -> str:
		self._counter += 1
		return f"t{self._counter}"

	def add_root(self, content: str, metadata: Optional[dict[str, Any]] = None) -> ThoughtNode:
		node = ThoughtNode(
			node_id=self._next_id(),
			content=content,
			depth=0,
			parent_id=None,
			metadata=metadata or {},
		)
		self.nodes[node.node_id] = node
		return node

	def add_node(
		self,
		content: str,
		parent_id: str,
		depth: int,
		relation: str = "expands",
		metadata: Optional[dict[str, Any]] = None,
	) -> ThoughtNode:
		if parent_id not in self.nodes:
			raise KeyError(f"Parent node `{parent_id}` does not exist in ThoughtGraph")
		node = ThoughtNode(
			node_id=self._next_id(),
			content=content,
			depth=depth,
			parent_id=parent_id,
			metadata=metadata or {},
		)
		self.nodes[node.node_id] = node
		self.edges.append(ThoughtEdge(source_id=parent_id, target_id=node.node_id, relation=relation))
		return node

	def update_score(self, node_id: str, score: float, rationale: str = "") -> None:
		node = self.nodes[node_id]
		node.score = float(max(0.0, min(1.0, score)))
		node.rationale = rationale

	def children(self, node_id: str) -> list[ThoughtNode]:
		child_ids = [edge.target_id for edge in self.edges if edge.source_id == node_id]
		return [self.nodes[cid] for cid in child_ids]

	def leaves(self) -> list[ThoughtNode]:
		parent_ids = {edge.source_id for edge in self.edges}
		return [node for node in self.nodes.values() if node.node_id not in parent_ids]

	def top_k_leaves(self, k: int) -> list[ThoughtNode]:
		leaves = self.leaves()
		return sorted(leaves, key=lambda n: n.score, reverse=True)[:k]

	def best_node(self) -> Optional[ThoughtNode]:
		if not self.nodes:
			return None
		return max(self.nodes.values(), key=lambda n: n.score)

	def path_to(self, node_id: str) -> list[ThoughtNode]:
		if node_id not in self.nodes:
			return []
		path: list[ThoughtNode] = []
		cursor = self.nodes[node_id]
		while True:
			path.append(cursor)
			if cursor.parent_id is None:
				break
			cursor = self.nodes[cursor.parent_id]
		return list(reversed(path))

	def as_dict(self) -> dict[str, Any]:
		return {
			"nodes": [
				{
					"node_id": node.node_id,
					"content": node.content,
					"depth": node.depth,
					"parent_id": node.parent_id,
					"score": node.score,
					"rationale": node.rationale,
					"metadata": node.metadata,
				}
				for node in self.nodes.values()
			],
			"edges": [
				{
					"source_id": edge.source_id,
					"target_id": edge.target_id,
					"relation": edge.relation,
				}
				for edge in self.edges
			],
		}
