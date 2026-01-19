#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
from typing import cast

# Add parent directory to path to allow imports when running script directly
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    args = parser.parse_args()

    match cast(str, args.command).lower():
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
