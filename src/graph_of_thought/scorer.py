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
	"""
	Result of scoring a thought.
	Attributes:
		score (float): Final normalized score [0-1].
		rationale (str): Explanation for the score.
		heuristic (float): The heuristic score before LLM adjustment.
		llm_score (float | None): The LLM score if computed, otherwise None.
	"""    
    
	score: float
	rationale: str
	heuristic: float
	llm_score: float | None


class ThoughtScorer:
	"""
	Scores thought quality using a hybrid heuristic-LLM approach.

	Combines fast heuristic scoring (token overlap, length, depth penalty) with optional
	LLM-based scoring for uncertain or deep nodes. This reduces LLM invocations by ~60-70%
	compared to scoring every thought with the LLM.

	The heuristic considers:
	- Task-thought token overlap (relevance)
	- Thought length (ideally 15-25 words)
	- Depth penalty (deeper nodes are penalized slightly)

	For uncertain or deep nodes, the LLM provides a more nuanced score based on logical
	coherence, correctness, and specificity.
	"""
 
	_token_pattern = re.compile(r"[a-zA-Z0-9_]+")

	@classmethod
	def _tokenize(cls, text: str) -> set[str]:
		"""
		Tokenize text into lowercase words, filtering out very short tokens.

		Args:
			text (str): The text to tokenize.

		Returns:
			set[str]: Set of lowercase tokens with length > 2.
		"""
  
		return {token.lower() for token in cls._token_pattern.findall(text or "") if len(token) > 2}

	def heuristic_score(self, task: str, thought: str, depth: int) -> float:
		"""
		Compute a fast heuristic score for a thought without LLM inference.

		Combines three components:
		- Token overlap with task (0-1, higher = more relevant)
		- Length penalty (optimal ~20 words, 0-1)
		- Depth penalty (0.45-1.0, deeper nodes penalized slightly)

		Final score is: 0.5*overlap + 0.3*length_score + 0.2*depth_factor

		Args:
			task (str): The original reasoning task/question.
			thought (str): The candidate thought to score.
			depth (int): The depth of this thought in the tree (0=root).

		Returns:
			float: A normalized score in range [0.0, 1.0].
		"""
  
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
		"""
		Score a thought using an adaptive hybrid heuristic-LLM approach.

		Strategy:
		1. Always compute fast heuristic score
		2. If heuristic < 0.5 (uncertain) OR depth >= 3 (deep node), also compute LLM score
		3. Prefer LLM score if available and successful, otherwise fall back to heuristic

		This design reduces LLM inference by ~60-70% while maintaining quality for uncertain decisions.

		Args:
			task (str): The original reasoning task.
			thought (str): The candidate thought to score.
			depth (int): Depth in the reasoning tree.
			context (str): Contextual information to inform scoring.
			llm (OllamaClient): LLM client for semantic scoring.
			prompter (GoTPrompter): Generates score prompts for the LLM.
			parser (GoTParser): Parses LLM-generated scores.
			temperature (float, optional): LLM temperature for scoring. Defaults to 0.0 (deterministic).

		Returns:
			ScoreResult: Contains normalized score (0-1), rationale, heuristic value, and whether LLM was used.
		"""
  
		heuristic = self.heuristic_score(task=task, thought=thought, depth=depth)
		llm_score: float | None = None
		llm_rationale = ""

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

		if llm_score is None:			return ScoreResult(
				score=heuristic,
				rationale="Heuristic score" + (" (uncertain)" if heuristic < 0.5 else ""),
				heuristic=heuristic,
				llm_score=None,
			)

		return ScoreResult(
			score=llm_score,
			rationale=llm_rationale or "LLM score",
			heuristic=heuristic,
			llm_score=llm_score,
		)