from __future__ import annotations

import json
import re
from typing import Any

class GoTParser:
	"""
	Parses structured JSON outputs from the Graph of Thought LLM responses.

	This parser handles the extraction of subproblems, candidate thoughts, and scores
	from LLM-generated responses. It is robust against various formatting variations,
	including markdown code blocks, whitespace inconsistencies, and malformed JSON.

	The parser implements a resilient candidate-generation approach: if the primary
	JSON parsing fails, it attempts multiple fallback strategies (e.g., extracting
	JSON from code blocks, searching for JSON-like structures).

	Example:
		parser = GoTParser()
		raw_response = '''```json
		{"subproblems": ["How to X?", "What about Y?"]}
		```'''
		subproblems = parser.parse_subproblems(raw_response)

	Raises:
		ValueError: If the parsed JSON does not contain the expected structure for
		            subproblems, candidates, or scores.
	"""    
    
	_code_block_pattern = re.compile(r"```(?:json)?\s*(.*?)```", flags=re.IGNORECASE | re.DOTALL)
	_json_block_pattern = re.compile(r"(\{.*\}|\[.*\])", flags=re.DOTALL)

	def _try_parse_json(self, raw: str) -> Any:
		"""
		Tries to parse a JSON string, handling code blocks and other formatting.

		Args:
			raw (str): The raw string to parse.

		Returns:
			Any: The parsed JSON object or None if parsing fails.
		"""		
     
		candidate = (raw or "").strip()
		if not candidate:
			return None

		for maybe in self._json_candidates(candidate):
			try:
				return json.loads(maybe)
			except json.JSONDecodeError:
				continue
		return None

	def _json_candidates(self, raw: str) -> list[str]:
		"""
		Generates a list of potential JSON strings from the raw input.

		Args:
			raw (str): The raw string to search for JSON candidates.

		Returns:
			list[str]: A list of potential JSON strings.
		"""		
     
		candidates = [raw]
		for match in self._code_block_pattern.findall(raw):
			candidates.append(match.strip())
		json_match = self._json_block_pattern.search(raw)
		if json_match:
			candidates.append(json_match.group(1).strip())

		seen: set[str] = set()
		ordered: list[str] = []
		for item in candidates:
			if item and item not in seen:
				seen.add(item)
				ordered.append(item)
		return ordered

	@staticmethod
	def _dedupe_keep_order(values: list[str]) -> list[str]:
		"""
		Removes duplicates from a list while preserving the original order.

		Args:
			values (list[str]): The list of strings to deduplicate.

		Returns:
			list[str]: The deduplicated list of strings.
		"""     
     
		seen: set[str] = set()
		out: list[str] = []
		for value in values:
			key = value.strip().lower()
			if not key or key in seen:
				continue
			seen.add(key)
			out.append(value.strip())
		return out

	def parse_subproblems(self, raw: str, max_items: int = 4) -> list[str]:
		"""
		Parses the subproblems from the raw input.

		Args:
			raw (str): The raw string to parse.
			max_items (int, optional): The maximum number of subproblems to return. Defaults to 4.

		Raises:
			ValueError: If the parsed JSON is invalid or does not contain the expected structure.

		Returns:
			list[str]: The list of parsed subproblems.
		"""     
     
		payload = self._try_parse_json(raw)
		values: list[str] = []

		if isinstance(payload, dict):
			data = payload.get("subproblems")
			if isinstance(data, list):
				values = [str(v).strip() for v in data if str(v).strip()]
		elif isinstance(payload, list):
			values = [str(v).strip() for v in payload if str(v).strip()]

		if not values:
			raise ValueError("Failed to parse GoT subproblems JSON response.")
		return self._dedupe_keep_order(values)[:max_items]

	def parse_candidates(self, raw: str, max_items: int = 4) -> list[str]:
		"""
		Parses candidate thoughts from a raw LLM response.

		Searches for a JSON structure containing candidate thoughts under common keys:
		"candidates", "thoughts", "steps", or "options". Falls back to treating the
		parsed JSON as a list if no recognized key is found.

		Args:
			raw (str): The raw LLM response to parse.
			max_items (int, optional): Maximum number of candidates to return. Defaults to 4.

		Returns:
			list[str]: Deduplicated list of candidate thoughts, up to max_items in length.

		Raises:
			ValueError: If no valid candidates are found in the JSON structure.
		"""
  
		payload = self._try_parse_json(raw)
		values: list[str] = []

		if isinstance(payload, dict):
			for key in ("candidates", "thoughts", "steps", "options"):
				data = payload.get(key)
				if isinstance(data, list):
					values = [str(v).strip() for v in data if str(v).strip()]
					if values:
						break
		elif isinstance(payload, list):
			values = [str(v).strip() for v in payload if str(v).strip()]

		if not values:
			raise ValueError("Failed to parse GoT candidate thoughts JSON response.")
		return self._dedupe_keep_order(values)[:max_items]

	def parse_score(self, raw: str) -> tuple[float, str]:
		"""
		Parses a quality score and rationale from a raw LLM response.

		Expects a JSON object with "score" and optional "rationale" keys.
		Handles both 0-1 and 0-10 score ranges, automatically normalizing to 0-1.
		Scores are clamped to the valid range [0.0, 1.0].

		Args:
			raw (str): The raw LLM response containing the score and rationale.

		Returns:
			tuple[float, str]: A tuple of (normalized_score, rationale_text).
			                   Score is in range [0.0, 1.0].
			                   Rationale defaults to "Score rationale not provided" if missing.

		Raises:
			ValueError: If no valid score is found in the JSON structure.

		Example:
			raw = '{"score": 8.5, "rationale": "Logically sound"}'
			score, rationale = parser.parse_score(raw)
			# score = 0.85, rationale = "Logically sound"
		"""
  
		payload = self._try_parse_json(raw)
		score = None
		rationale = ""

		if isinstance(payload, dict):
			if "score" in payload:
				try:
					score = float(payload["score"])
				except (TypeError, ValueError):
					score = None
			if "rationale" in payload:
				rationale = str(payload.get("rationale", "")).strip()

		if score is None:
			raise ValueError("Failed to parse GoT score JSON response.")

		if score > 1.0 and score <= 10.0:
			score = score / 10.0
		score = max(0.0, min(1.0, score))

		if not rationale:
			rationale = "Score rationale not provided"
		return score, rationale