from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from .graph import ThoughtGraph, ThoughtNode

ExpandFn = Callable[[ThoughtNode, int], Sequence[str]]
ScoreFn = Callable[[ThoughtNode], tuple[float, str]]


@dataclass
class SearchConfig:
	max_depth: int = 3
	beam_width: int = 3
	branch_factor: int = 3


class GoTSearcher:
	"""Beam-search style traversal through generated thought graph branches."""

	def __init__(self, config: SearchConfig) -> None:
		self.config = config

	def run(
		self,
		graph: ThoughtGraph,
		initial_frontier: list[ThoughtNode],
		expand_fn: ExpandFn,
		score_fn: ScoreFn,
	) -> list[ThoughtNode]:
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
