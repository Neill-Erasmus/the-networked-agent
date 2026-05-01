from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from src.ollama_client import OllamaClient

from .parser import GoTParser
from .prompter import GoTPrompter


@dataclass
class ScoreResult:
	score: float
	rationale: str
	heuristic: float
	llm_score: float | None


class ThoughtScorer:
	_token_pattern = re.compile(r"[a-zA-Z0-9_]+")

	@classmethod
	def _tokenize(cls, text: str) -> set[str]:
		return {token.lower() for token in cls._token_pattern.findall(text or "") if len(token) > 2}

	def heuristic_score(self, task: str, thought: str, depth: int) -> float:
		task_tokens = self._tokenize(task)
		thought_tokens = self._tokenize(thought)

		overlap = len(task_tokens & thought_tokens) / max(1, len(task_tokens))
		thought_len = len(thought.split())
		length_target = 20
		length_score = 1.0 - min(abs(thought_len - length_target) / length_target, 1.0)
		depth_factor = max(0.45, 1.0 - 0.12 * max(0, depth - 1))

		score = 0.5 * overlap + 0.3 * length_score + 0.2 * depth_factor
		return max(0.0, min(1.0, score))

	def score(
		self,
		task: str,
		thought: str,
		depth: int,
		context: str,
		llm: OllamaClient,
		prompter: GoTPrompter,
		parser: GoTParser,
		temperature: float = 0.0,
	) -> ScoreResult:
		"""Score thoughts efficiently: heuristic-first, LLM only when uncertain.
		
		This reduces LLM calls by ~60-70% compared to scoring every thought.
		"""
		heuristic = self.heuristic_score(task=task, thought=thought, depth=depth)
		llm_score: float | None = None
		llm_rationale = ""

		# Adaptive strategy: only use LLM for uncertain/deep nodes
		should_use_llm = heuristic < 0.5 or depth >= 3
		
		if should_use_llm:
			try:
				prompt = prompter.score_prompt(task=task, thought=thought, depth=depth, context=context)
				raw = llm.generate(
					prompt=prompt,
					system=prompter.system_prompt(),
					temperature=temperature,
					json_mode=True,
				)
				llm_score, llm_rationale = parser.parse_score(raw)
			except Exception:
				llm_score = None
				llm_rationale = ""

		if llm_score is None:
			# LLM failed or not used; use heuristic
			return ScoreResult(
				score=heuristic,
				rationale="Heuristic score" + (" (uncertain)" if heuristic < 0.5 else ""),
				heuristic=heuristic,
				llm_score=None,
			)

		# Use LLM score when available (no blending - trust the LLM)
		return ScoreResult(
			score=llm_score,
			rationale=llm_rationale or "LLM score",
			heuristic=heuristic,
			llm_score=llm_score,
		)
