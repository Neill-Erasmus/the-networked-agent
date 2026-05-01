from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from .graph import ThoughtGraph, ThoughtNode

ExpandFn = Callable[[ThoughtNode, int], Sequence[str]]
ScoreFn = Callable[[ThoughtNode], tuple[float, str]]

@dataclass
class SearchConfig:
	"""
	Configuration for Graph of Thought search.
	Attributes:
		max_depth (int): Maximum depth of the search tree.
		beam_width (int): Number of top nodes to keep at each level.
		branch_factor (int): Number of expansions to generate per node.
	"""    
    
	max_depth: int = 3
	beam_width: int = 3
	branch_factor: int = 3


class GoTSearcher:
	"""
	Performs beam-search traversal through a thought graph.

	Uses a breadth-first beam search strategy to explore the most promising reasoning
	paths. At each depth, the top-k (beam_width) nodes are expanded into new candidates,
	scored, and the best scorers become the frontier for the next iteration.

	This approach balances exploration of multiple paths with computational efficiency
	by pruning low-scoring branches early.

	Attributes:
		config (SearchConfig): Configuration controlling search parameters.
	"""

	def __init__(self, config: SearchConfig) -> None:
		self.config = config

	def run(
		self,
		graph: ThoughtGraph,
		initial_frontier: list[ThoughtNode],
		expand_fn: ExpandFn,
		score_fn: ScoreFn,
	) -> list[ThoughtNode]:
		"""
		Execute beam search through the thought graph.

		Iteratively expands the frontier of highest-scoring nodes, scores new candidates,
		and maintains the top-k nodes as the next frontier. Continues until max_depth is
		reached or no more candidates can be generated.

		Args:
			graph (ThoughtGraph): The graph to store generated nodes and edges.
			initial_frontier (list[ThoughtNode]): Starting nodes for the search (usually root).
			expand_fn (ExpandFn): Function that generates candidate thoughts from a node.
			                      Signature: (node, branch_factor) -> Sequence[str]
			score_fn (ScoreFn): Function that evaluates and scores a thought node.
			                    Signature: (node) -> (score: float, rationale: str)

		Returns:
			list[ThoughtNode]: The top-k leaves by score, representing the best reasoning paths found.
		"""
  
		frontier = sorted(initial_frontier, key=lambda n: n.score, reverse=True)[: self.config.beam_width]

		for _ in range(self.config.max_depth):
			if not frontier:
				break
			candidate_nodes: list[ThoughtNode] = []

			for node in frontier[: self.config.beam_width]:
				if node.depth >= self.config.max_depth:
					continue

				expansions = expand_fn(node, self.config.branch_factor)
				for thought in list(expansions)[: self.config.branch_factor]:
					child = graph.add_node(
						content=thought,
						parent_id=node.node_id,
						depth=node.depth + 1,
						relation="expands",
					)
					score, rationale = score_fn(child)
					graph.update_score(child.node_id, score=score, rationale=rationale)
					candidate_nodes.append(child)

			if not candidate_nodes:
				break

			candidate_nodes.sort(key=lambda n: n.score, reverse=True)
			frontier = candidate_nodes[: self.config.beam_width]

		if frontier:
			return sorted(frontier, key=lambda n: n.score, reverse=True)
		return graph.top_k_leaves(self.config.beam_width)