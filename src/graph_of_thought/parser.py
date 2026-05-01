from __future__ import annotations

import json
import re
from typing import Any


class GoTParser:
	_code_block_pattern = re.compile(r"```(?:json)?\s*(.*?)```", flags=re.IGNORECASE | re.DOTALL)
	_json_block_pattern = re.compile(r"(\{.*\}|\[.*\])", flags=re.DOTALL)

	def _try_parse_json(self, raw: str) -> Any:
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
