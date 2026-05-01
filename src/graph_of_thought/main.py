from __future__ import annotations

import argparse
import os

from src.ollama_client import OllamaClient, OllamaConfig, OllamaError

from .controller import GoTConfig, GraphOfThoughtController


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Run Graph of Thought reasoning over a task.")
	parser.add_argument("--task", type=str, default="", help="Task/question to solve")
	parser.add_argument("--context", type=str, default="", help="Optional retrieval context")
	parser.add_argument("--context-file", type=str, default="", help="Optional text file with context")
	parser.add_argument("--base-url", type=str, default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"), help="Ollama URL")
	parser.add_argument("--chat-model", type=str, default=os.getenv("OLLAMA_CHAT_MODEL", "llama3:latest"), help="Ollama chat model")
	parser.add_argument("--embed-model", type=str, default=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"), help="Ollama embedding model")
	parser.add_argument("--max-depth", type=int, default=3)
	parser.add_argument("--beam-width", type=int, default=3)
	parser.add_argument("--branch-factor", type=int, default=3)
	parser.add_argument("--temperature", type=float, default=0.2)
	return parser


def main() -> None:
	args = build_parser().parse_args()

	task = args.task.strip() or input("Task: ").strip()
	context = args.context
	if args.context_file:
		with open(args.context_file, "r", encoding="utf-8") as infile:
			context = (context + "\n" + infile.read()).strip()

	llm = OllamaClient(
		OllamaConfig(
			base_url=args.base_url,
			chat_model=args.chat_model,
			embedding_model=args.embed_model,
		)
	)

	if not llm.healthcheck():
		raise OllamaError("Ollama server is not reachable. Start it with `ollama serve`.")

	llm.assert_model_available(llm.config.chat_model)

	controller = GraphOfThoughtController(
		llm=llm,
		config=GoTConfig(
			max_depth=args.max_depth,
			beam_width=args.beam_width,
			branch_factor=args.branch_factor,
			temperature=args.temperature,
		),
	)

	result = controller.solve(task=task, context=context)
	print("\n=== Graph of Thought Result ===")
	print(result.answer)
	print("\n--- Reasoning Path ---")
	for idx, step in enumerate(result.reasoning_path, start=1):
		print(f"{idx}. {step}")


if __name__ == "__main__":
	main()
