from __future__ import annotations

import argparse
import os

from src.ollama_client import OllamaClient, OllamaConfig, OllamaError

from .engine import GraphRAGConfig, GraphRAGEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run simple local GraphRAG with Ollama.")
    parser.add_argument("--base-url", type=str, default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    parser.add_argument("--chat-model", type=str, default=os.getenv("OLLAMA_CHAT_MODEL", "llama3:latest"))
    parser.add_argument("--embed-model", type=str, default=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))
    parser.add_argument("--store-path", type=str, default="data/graphrag_store.json")
    parser.add_argument("--query", type=str, default="")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--hops", type=int, default=1)
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
        raise OllamaError("Ollama server is not reachable. Start it with `ollama serve`.")

    llm.assert_model_available(llm.config.chat_model)
    llm.assert_model_available(llm.config.embedding_model)

    engine = GraphRAGEngine(
        llm=llm,
        config=GraphRAGConfig(
            store_path=args.store_path,
            default_top_k=args.top_k,
            default_hops=args.hops,
        ),
    )

    if args.query.strip():
        answer, retrieval = engine.answer(args.query, top_k=args.top_k, hops=args.hops)
        print("\n=== GraphRAG Answer ===")
        print(answer)
        print("\n=== Retrieved Chunks ===")
        for hit in retrieval.hits:
            print(f"{hit.chunk_id} (score={hit.score:.3f}, source={hit.source}, depth={hit.depth})")


if __name__ == "__main__":
    main()