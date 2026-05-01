from __future__ import annotations

import argparse
import os

from src.networked_agent import NetworkedAgent, NetworkedAgentConfig
from src.ollama_client import OllamaClient, OllamaConfig, OllamaError


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Networked Agent powered by GoT + GraphRAG on local Ollama.")
	parser.add_argument("--base-url", type=str, default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
	parser.add_argument("--chat-model", type=str, default=os.getenv("OLLAMA_CHAT_MODEL", "llama3:latest"))
	parser.add_argument("--embed-model", type=str, default=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest"))
	parser.add_argument("--store-path", type=str, default="data/graphrag_store.json")

	parser.add_argument("--ask", type=str, default="")
	parser.add_argument("--ingest-file", type=str, default="", help="Ingest one .txt or .json file into the GraphRAG store")
	parser.add_argument("--interactive", action="store_true")
	parser.add_argument("--visualization-dir", type=str, default=os.getenv("AGENT_VISUALIZATION_DIR", "data/visualizations"))
	parser.add_argument("--no-visuals", action="store_true", help="Disable per-query visualization artifacts")

	parser.add_argument("--got-max-depth", type=int, default=3)
	parser.add_argument("--got-beam-width", type=int, default=3)
	parser.add_argument("--got-branch-factor", type=int, default=3)
	parser.add_argument("--got-temperature", type=float, default=0.2)
	parser.add_argument("--rag-top-k", type=int, default=4)
	parser.add_argument("--rag-hops", type=int, default=1)
	return parser


def main() -> None:
	args = build_parser().parse_args()

	llm = OllamaClient(
		OllamaConfig(
			base_url=args.base_url,
			chat_model=args.chat_model,
			embedding_model=args.embed_model,
		)
	)

	if not llm.healthcheck():
		raise OllamaError(
			"Ollama server is not reachable. Start it with `ollama serve` and "
			"ensure the URL is correct."
		)

	llm.assert_model_available(llm.config.chat_model)
	llm.assert_model_available(llm.config.embedding_model)

	agent = NetworkedAgent(
		llm=llm,
		config=NetworkedAgentConfig(
			store_path=args.store_path,
			got_max_depth=args.got_max_depth,
			got_beam_width=args.got_beam_width,
			got_branch_factor=args.got_branch_factor,
			got_temperature=args.got_temperature,
			rag_top_k=args.rag_top_k,
			rag_hops=args.rag_hops,
			save_visualizations=not args.no_visuals,
			visualization_dir=args.visualization_dir,
		),
	)

	if args.ingest_file.strip():
		doc_id = agent.ingest_file(args.ingest_file)
		print("\n=== Ingestion Complete ===")
		print(f"Document ID: {doc_id}")
		print(f"Store path: {args.store_path}")

	if args.ask.strip():
		turn = agent.think_and_answer(args.ask)
		print("\n=== Agent Answer ===")
		print(turn.answer)
		if turn.retrieved_chunk_ids:
			print("\nRetrieved chunks: " + ", ".join(turn.retrieved_chunk_ids))
		if turn.reasoning_path:
			print("\nReasoning path:")
			for idx, step in enumerate(turn.reasoning_path, start=1):
				print(f"{idx}. {step}")
		if turn.visualization_html:
			print("\nVisualization file:")
			print(turn.visualization_html)

	if args.interactive or (not args.ask.strip() and not args.ingest_file.strip()):
		agent.run_interactive()


if __name__ == "__main__":
	main()
