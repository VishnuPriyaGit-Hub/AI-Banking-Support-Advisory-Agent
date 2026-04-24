from __future__ import annotations

import argparse
import json

from app.rag.ingest import SimpleRAGIngestor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk docs, embed them, and load them into Zilliz Cloud / Milvus.")
    parser.add_argument("--chunk-size", type=int, default=800, help="Chunk size in characters.")
    parser.add_argument("--chunk-overlap", type=int, default=120, help="Overlap between chunks in characters.")
    parser.add_argument("--batch-size", type=int, default=16, help="Embedding batch size.")
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop the collection before recreating it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ingestor = SimpleRAGIngestor()
    summary = ingestor.run(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        drop_existing=args.drop_existing,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
