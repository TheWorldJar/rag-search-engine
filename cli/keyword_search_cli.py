#!/usr/bin/env python3

import argparse
import json


def keywordSearch(query: str) -> None:
    found = []
    with open("./data/movies.json", "r", encoding="utf-8") as f:
        movies = json.load(f)
        for movie in movies["movies"]:
            if query in movie["title"]:
                found.append(movie["title"])

    for i in range(len(found)):
        print(f"{i}. {found[i]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print("Searching for:", args.query)
            keywordSearch(args.query)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
