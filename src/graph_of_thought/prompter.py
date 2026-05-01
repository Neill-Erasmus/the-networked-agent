from __future__ import annotations


class GoTPrompter:
	"""
	Generates prompts for Graph of Thought reasoning phases.

	Produces structured prompts for:
	- Decomposition: Breaking a task into subproblems
	- Expansion: Generating candidate next thoughts
	- Scoring: Evaluating thought quality
	- Synthesis: Assembling final answer from reasoning path

	Handles context truncation to fit within token budgets and enforces JSON output format
	for reliable parsing of LLM responses.

	Attributes:
		max_context_chars (int): Maximum context length before truncation. Defaults to 3500.
	"""

	def __init__(self, max_context_chars: int = 3500) -> None:
		self.max_context_chars = max_context_chars

	def system_prompt(self) -> str:
		"""
		Return the system prompt for GoT reasoning phases.

		Instructs the LLM to act as a deliberate reasoning assistant following Graph of Thought principles,
		emphasizing clear and structured output.

		Returns:
			str: System prompt for reasoning tasks.
		"""
  
		return (
			"You are a deliberate reasoning assistant that follows Graph of Thought principles. "
			"Generate clear, concrete, high-value reasoning steps and strictly follow requested output format."
		)

	def _trim_context(self, context: str) -> str:
		"""
		Truncate context to fit within token budget.

		Args:
			context (str): Raw context string, potentially very long.

		Returns:
			str: Trimmed context, safe for prompt inclusion.
		"""
  
		context = (context or "").strip()
		if len(context) <= self.max_context_chars:
			return context
		return context[: self.max_context_chars] + "\n[Context truncated]"

	def decompose_prompt(self, task: str, context: str, max_subproblems: int) -> str:
		"""
		Generate a prompt for task decomposition into subproblems.

		Args:
			task (str): The high-level task to decompose.
			context (str): Background context for the task.
			max_subproblems (int): Target number of subproblems to generate.

		Returns:
			str: Prompt requesting JSON with "subproblems" and "notes" keys.
		"""
  
		safe_context = self._trim_context(context)
		return f"""
				Task:
				{task}

				Context:
				{safe_context or "No external context provided."}

				Break the task into {max_subproblems} concrete sub-problems that can be solved independently and later combined.
				Return ONLY valid JSON using this exact schema:
				{{
				"subproblems": ["..."],
				"notes": "short planning rationale"
				}}
				""".strip()

	def expand_prompt(self, task: str, parent_thought: str, context: str, branch_factor: int) -> str:
		"""
		Generate a prompt for expanding a thought node.

		Args:
			task (str): The original task/question.
			parent_thought (str): The current thought node to expand from.
			context (str): Background context.
			branch_factor (int): Number of distinct candidate thoughts to generate.

		Returns:
			str: Prompt requesting JSON with "candidates" and "notes" keys.
		"""
  
		safe_context = self._trim_context(context)
		return f"""
				Global task:
				{task}

				Current thought to expand:
				{parent_thought}

				Context:
				{safe_context or "No external context provided."}

				Generate {branch_factor} distinct, high-quality next thoughts that improve correctness or completeness.
				Each thought should be concise and actionable.
				Return ONLY valid JSON:
				{{
				"candidates": ["...", "..."],
				"notes": "short note"
				}}
				""".strip()

	def score_prompt(self, task: str, thought: str, depth: int, context: str) -> str:
		"""
		Generate a prompt for scoring a candidate thought.

		Args:
			task (str): The original task/question.
			thought (str): The candidate thought to evaluate.
			depth (int): The depth in the reasoning tree.
			context (str): Background context.

		Returns:
			str: Prompt requesting JSON with "score" (0-1) and "rationale" keys.
		"""
  
		safe_context = self._trim_context(context)
		return f"""
				Task:
				{task}

				Candidate thought (depth={depth}):
				{thought}

				Context:
				{safe_context or "No external context provided."}

				Score this thought from 0.0 to 1.0 based on:
				1) relevance to the task,
				2) correctness,
				3) specificity,
				4) contribution beyond prior steps.

				Return ONLY valid JSON:
				{{
				"score": 0.0,
				"rationale": "one short reason"
				}}
				""".strip()

	def synthesis_prompt(self, task: str, reasoning_steps: list[str], context: str) -> str:
		"""
		Generate a prompt for synthesizing a final answer from a reasoning path.

		Args:
			task (str): The original task/question to answer.
			reasoning_steps (list[str]): The best reasoning path found by search.
			context (str): Background context to inform the answer.

		Returns:
			str: Prompt instructing the LLM to synthesize a coherent final answer.
		"""
  
		safe_context = self._trim_context(context)
		rendered_steps = "\n".join(f"{idx}. {step}" for idx, step in enumerate(reasoning_steps, start=1))
		return f"""
				You must answer the task using the reasoning path and context below.

				Task:
				{task}

				Reasoning path:
				{rendered_steps or "No steps available."}

				Context:
				{safe_context or "No external context provided."}

				Write a direct, coherent final answer. If context is insufficient, say what is missing.
				""".strip()