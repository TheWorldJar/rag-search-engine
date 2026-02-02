#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
from typing import cast

# Add parent directory to path to allow imports when running script directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.lib.semantic_search import embed_text, verify_model


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    _ = subparsers.add_parser("verify", description="Verify Semantic Search Model")
    embed_text_parser = subparsers.add_parser("embed_text", description="Embed Text")
    _ = embed_text_parser.add_argument("text", type=str, help="Text to embed")

    args = parser.parse_args()

    match cast(str, args.command).lower():
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(cast(str, args.text))
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
