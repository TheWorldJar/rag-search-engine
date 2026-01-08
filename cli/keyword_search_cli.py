#!/usr/bin/env python3

import argparse
import json
import re
import string
from typing import TypedDict, cast

from nltk.stem import PorterStemmer


class Movie(TypedDict):
    id: int
    title: str
    description: str


def tokenize(query: str) -> list[str]:
    translator = str.maketrans("", "", string.punctuation)
    return query.translate(translator).lower().split(" ")


def stop(query: list[str]) -> list[str]:
    f = open("./data/stopwords.txt", "r", encoding="utf-8")
    stop_words = f.read().splitlines()
    f.close()
    return [token for token in query if token not in stop_words]


def stem(query: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in query]


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

    args = parser.parse_args()

    match cast(str, args.command).lower():
        case "search":
            print("Searching for:", cast(str, args.query))
            keywordSearch(cast(str, args.query))
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
