#!/usr/bin/env python3

import argparse
import json
import re
import string
import sys
from pathlib import Path
from typing import cast

# Add parent directory to path to allow imports when running script directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.inverted_index import InvertedIndex
from cli.token_utils import Movie, stem, stop, tokenize


def keywordSearch(query: str) -> None:
    found: list[str] = []
    translator = str.maketrans("", "", string.punctuation)
    stemmed_query = stem(stop(tokenize(query)))

    with open("./data/movies.json", "r", encoding="utf-8") as f:
        movies: list[Movie] = cast(list[Movie], json.load(f)["movies"])
        for movie in movies:
            translated_movie_title = movie["title"].translate(translator).lower()
            for q_token in stemmed_query:
                pattern = re.compile(rf"{re.escape(q_token)}.*")
                if pattern.search(translated_movie_title):
                    found.append(movie["title"])
                    break
        f.close()

    for i in range(len(found)):
        print(f"{i}. {found[i]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    _ = search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build inverted index")

    args = parser.parse_args()
    index = InvertedIndex()

    match cast(str, args.command).lower():
        case "search":
            print("Searching for:", cast(str, args.query))
            keywordSearch(cast(str, args.query))
        case "build":
            index.build()
            index.save()
            print(f"First document for token 'meridia' = {index.get_documents('merida')[0]}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
