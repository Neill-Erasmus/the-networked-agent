from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from src.ollama_client import OllamaClient

from .graph import ThoughtGraph, ThoughtNode
from .parser import GoTParser
from .prompter import GoTPrompter
from .scorer import ThoughtScorer
from .search import GoTSearcher, SearchConfig

# Retrieval callback type: (task, context, depth) -> new_context
RetrievalFn = Callable[[str, str, int], str]


@dataclass
class GoTConfig:
	max_depth: int = 3
	beam_width: int = 3
	branch_factor: int = 3
	temperature: float = 0.2
	max_subproblems: int = 3


@dataclass
class GoTResult:
	task: str
	answer: str
	best_thought: str
	reasoning_path: list[str]
	graph_snapshot: dict[str, Any]
	retrieval_context: str


class GraphOfThoughtController:
	"""Coordinates decomposition, expansion, scoring, and final synthesis."""

	def __init__(self, llm: OllamaClient, config: Optional[GoTConfig] = None) -> None:
		self.llm = llm
		self.config = config or GoTConfig()
		self.prompter = GoTPrompter()
		self.parser = GoTParser()
		self.scorer = ThoughtScorer()
		self.searcher = GoTSearcher(
			SearchConfig(
				max_depth=self.config.max_depth,
				beam_width=self.config.beam_width,
				branch_factor=self.config.branch_factor,
			)
		)

	def _decompose(self, task: str, context: str) -> list[str]:
		prompt = self.prompter.decompose_prompt(
			task=task,
			context=context,
			max_subproblems=self.config.max_subproblems,
		)
		raw = self.llm.generate(
			prompt=prompt,
			system=self.prompter.system_prompt(),
			temperature=min(0.2, self.config.temperature),
			json_mode=True,
		)
		return self.parser.parse_subproblems(raw, max_items=self.config.max_subproblems)

	def _expand(self, task: str, node: ThoughtNode, context: str, branch_factor: int) -> list[str]:
		prompt = self.prompter.expand_prompt(
			task=task,
			parent_thought=node.content,
			context=context,
			branch_factor=branch_factor,
		)
		raw = self.llm.generate(
			prompt=prompt,
			system=self.prompter.system_prompt(),
			temperature=self.config.temperature,
			json_mode=True,
		)
		candidates = self.parser.parse_candidates(raw, max_items=branch_factor)
		return candidates[:branch_factor]

	def solve(self, task: str, context: str = "", retrieval_fn: Optional[RetrievalFn] = None) -> GoTResult:
		"""Solve task via GoT with optional dynamic context refinement.
		
		Args:
			task: The problem to solve
			context: Initial retrieval context
			retrieval_fn: Optional callback to refine context during reasoning.
				Called as retrieval_fn(task, current_context, depth) -> new_context
		"""
		graph = ThoughtGraph()
		root = graph.add_root(task)

		try:
			seeds = self._decompose(task=task, context=context)
		except Exception:
			seeds = [task]
		frontier: list[ThoughtNode] = []
		for seed in seeds[: self.config.beam_width]:
			node = graph.add_node(
				content=seed,
				parent_id=root.node_id,
				depth=1,
				relation="decomposes",
			)
			score_result = self.scorer.score(
				task=task,
				thought=node.content,
				depth=node.depth,
				context=context,
				llm=self.llm,
				prompter=self.prompter,
				parser=self.parser,
				temperature=0.0,
			)
			graph.update_score(node.node_id, score=score_result.score, rationale=score_result.rationale)
			frontier.append(node)

		# Mutable context that can be refined during search
		current_context = context

		def expand_fn(node: ThoughtNode, branch_factor: int) -> list[str]:
			return self._expand(task=task, node=node, context=current_context, branch_factor=branch_factor)

		def score_fn(node: ThoughtNode) -> tuple[float, str]:
			nonlocal current_context

			score_result = self.scorer.score(
				task=task,
				thought=node.content,
				depth=node.depth,
				context=current_context,
				llm=self.llm,
				prompter=self.prompter,
				parser=self.parser,
				temperature=0.0,
			)
			final_score = score_result.score
			rationale = score_result.rationale

			# Dynamic context refinement: if score is low and retrieval is available, refresh context.
			if final_score < 0.4 and retrieval_fn is not None and current_context:
				new_context = retrieval_fn(task, current_context, node.depth)
				if new_context and new_context != current_context:
					current_context = new_context
					score_result = self.scorer.score(
						task=task,
						thought=node.content,
						depth=node.depth,
						context=current_context,
						llm=self.llm,
						prompter=self.prompter,
						parser=self.parser,
						temperature=0.0,
					)
					final_score = score_result.score
					rationale = score_result.rationale + " (refined context)"
			
			return final_score, rationale

		finalists = self.searcher.run(
			graph=graph,
			initial_frontier=frontier,
			expand_fn=expand_fn,
			score_fn=score_fn,
		)
		best = finalists[0] if finalists else (graph.best_node() or root)
		path_nodes = graph.path_to(best.node_id)
		reasoning_path = [node.content for node in path_nodes if node.node_id != root.node_id]

		final_prompt = self.prompter.synthesis_prompt(
			task=task,
			reasoning_steps=reasoning_path,
			context=current_context,
		)
		answer = self.llm.generate(
			prompt=final_prompt,
			system=self.prompter.system_prompt(),
			temperature=self.config.temperature,
		)

		return GoTResult(
			task=task,
			answer=answer,
			best_thought=best.content,
			reasoning_path=reasoning_path,
			graph_snapshot=graph.as_dict(),
			retrieval_context=current_context,
		)
