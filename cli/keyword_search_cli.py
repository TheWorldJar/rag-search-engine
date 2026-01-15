#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
from typing import cast

# Add parent directory to path to allow imports when running script directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.inverted_index import InvertedIndex
from cli.token_utils import stem, stop, tokenize


def keywordSearch(query: str, index: InvertedIndex) -> None:
    found: list[int] = []
    stemmed_query = stem(stop(tokenize(query)))

    for token in stemmed_query:
        found.extend(index.get_documents(token))
        if len(found) >= 5:
            break

    found = found[:5]
    for i in range(len(found)):
        print(f"id:{index.docmap[found[i]]['id']} | title:{index.docmap[found[i]]['title']}")


def main() -> None:
    try:
        parser = argparse.ArgumentParser(description="Keyword Search CLI")
        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        search_parser = subparsers.add_parser("search", help="Search movies using BM25")
        _ = search_parser.add_argument("query", type=str, help="Search query")

        _ = subparsers.add_parser("build", help="Build inverted index")

        tf_parser = subparsers.add_parser("tf", help="Get token frequency")
        _ = tf_parser.add_argument("doc_id", type=int, help="Document ID")
        _ = tf_parser.add_argument("term", type=str, help="Term")

        idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency")
        _ = idf_parser.add_argument("term", type=str, help="Term")

        tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF")
        _ = tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
        _ = tfidf_parser.add_argument("term", type=str, help="Term")

        bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 Inverse Document Frequency")
        _ = bm25_idf_parser.add_argument("term", type=str, help="Term")

        args = parser.parse_args()
        index = InvertedIndex()

        match cast(str, args.command).lower():
            case "search":
                index.load()
                print("Searching for:", cast(str, args.query))
                keywordSearch(cast(str, args.query), index)
            case "build":
                index.build()
                index.save()
                print("Inverted index built and saved")
            case "tf":
                index.load()
                print("Getting token frequency for:", cast(str, args.doc_id), cast(str, args.term))
                print(index.get_tf(cast(int, args.doc_id), cast(str, args.term)))
            case "idf":
                index.load()
                print("Getting inverse document frequency for:", cast(str, args.term))
                idf = index.get_idf(cast(str, args.term))
                print(f"Inverse Document Frequency of '{cast(str, args.term)}': {idf:0.2f}")
            case "tfidf":
                index.load()
                print("Getting TF-IDF for:", cast(str, args.doc_id), cast(str, args.term))
                tfidf = index.get_tfidf(cast(int, args.doc_id), cast(str, args.term))
                print(f"TF-IDF of '{cast(str, args.term)}' in document {cast(str, args.doc_id)}: {tfidf:0.2f}")
            case "bm25idf":
                index.load()
                print("Getting BM25 Inverse Document Frequency for:", cast(str, args.term))
                bm25idf = index.get_bm25_idf(cast(str, args.term))
                print(f"BM25 Inverse Document Frequency of '{cast(str, args.term)}': {bm25idf:0.2f}")
            case _:
                parser.print_help()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
