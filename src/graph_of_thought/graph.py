from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ThoughtNode:
	"""
	Represents a single thought in the reasoning graph.

	Attributes:
		node_id (str): Unique identifier for this node.
		content (str): The actual thought text.
		depth (int): Depth in the tree (0 = root).
		parent_id (Optional[str]): ID of the parent node, None if root.
		score (float): Quality score [0-1]. Defaults to 0.0.
		rationale (str): Explanation for the score.
		metadata (dict[str, Any]): Arbitrary metadata attached to the node.
	"""
 
	node_id: str
	content: str
	depth: int
	parent_id: Optional[str]
	score: float = 0.0
	rationale: str = ""
	metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ThoughtEdge:
	"""
	Represents a directed edge between two thoughts.

	Attributes:
		source_id (str): ID of the source node.
		target_id (str): ID of the target node.
		relation (str): Type of relation (e.g., "expands"). Defaults to "expands".
	"""
 
	source_id: str
	target_id: str
	relation: str = "expands"


class ThoughtGraph:
	"""
	In-memory graph storage for thoughts and their relations.

	Maintains a collection of thought nodes and directed edges representing the reasoning
	process. Supports adding nodes, updating scores, querying paths, and serialization.

	This is the core data structure for Graph of Thought search.
	"""

	def __init__(self) -> None:
		self.nodes: dict[str, ThoughtNode] = {}
		self.edges: list[ThoughtEdge] = []
		self._counter = 0

	def _next_id(self) -> str:
		"""
		Generate the next unique node ID.

		Returns:
			str: A unique identifier like "t1", "t2", etc.
		"""
  
		self._counter += 1
		return f"t{self._counter}"

	def add_root(self, content: str, metadata: Optional[dict[str, Any]] = None) -> ThoughtNode:
		"""
		Add the root (initial) thought to the graph.

		Args:
			content (str): The root thought text.
			metadata (Optional[dict[str, Any]]): Optional metadata for the root.

		Returns:
			ThoughtNode: The created root node with depth=0 and parent_id=None.
		"""
  
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
		"""
		Add a child node to the graph with an edge from parent.

		Args:
			content (str): The thought text.
			parent_id (str): ID of the parent node (must exist).
			depth (int): Depth in tree (typically parent_depth + 1).
			relation (str, optional): Edge relation type. Defaults to "expands".
			metadata (Optional[dict[str, Any]]): Optional node metadata.

		Returns:
			ThoughtNode: The created node with edge from parent.

		Raises:
			KeyError: If parent_id does not exist in the graph.
		"""
  
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
		"""
		Update the score and rationale for a node.

		Args:
			node_id (str): ID of the node to update.
			score (float): New score value, will be clamped to [0.0, 1.0].
			rationale (str, optional): Explanation for the score.
		"""
  
		node = self.nodes[node_id]
		node.score = float(max(0.0, min(1.0, score)))
		node.rationale = rationale

	def children(self, node_id: str) -> list[ThoughtNode]:
		"""
		Get all direct child nodes of a given node.

		Args:
			node_id (str): ID of the parent node.

		Returns:
			list[ThoughtNode]: List of child nodes, empty if none.
		"""
  
		child_ids = [edge.target_id for edge in self.edges if edge.source_id == node_id]
		return [self.nodes[cid] for cid in child_ids]

	def leaves(self) -> list[ThoughtNode]:
		"""
		Get all leaf nodes (nodes with no children).

		Returns:
			list[ThoughtNode]: All nodes that have not been expanded.
		"""
  
		parent_ids = {edge.source_id for edge in self.edges}
		return [node for node in self.nodes.values() if node.node_id not in parent_ids]

	def top_k_leaves(self, k: int) -> list[ThoughtNode]:
		"""
		Get the top-k leaf nodes by score.

		Args:
			k (int): Number of top leaves to return.

		Returns:
			list[ThoughtNode]: Up to k leaf nodes sorted by score descending.
		"""
  
		leaves = self.leaves()
		return sorted(leaves, key=lambda n: n.score, reverse=True)[:k]

	def best_node(self) -> Optional[ThoughtNode]:
		"""
		Get the highest-scoring node in the entire graph.

		Returns:
			Optional[ThoughtNode]: The best node by score, or None if graph is empty.
		"""
  
		if not self.nodes:
			return None
		return max(self.nodes.values(), key=lambda n: n.score)

	def path_to(self, node_id: str) -> list[ThoughtNode]:
		"""
		Get the complete path from root to a specific node.

		Args:
			node_id (str): ID of the target node.

		Returns:
			list[ThoughtNode]: Ordered list from root to target, empty if node not found.
		"""
  
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
		"""
		Serialize the graph to a dictionary.

		Returns:
			dict[str, Any]: Dictionary with "nodes" and "edges" keys containing serialized data.
		"""
  
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