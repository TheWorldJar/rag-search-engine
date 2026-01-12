import json
import os
import sys
from collections import Counter
from pathlib import Path
from pickle import dump, load
from typing import cast

# Add parent directory to path to allow imports when running script directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.token_utils import Movie, stem, stop, tokenize


class InvertedIndex:
    index: dict[str, set[str]]
    docmap: dict[str, Movie]
    term_freq: dict[str, Counter[str]]

    def __init__(self) -> None:
        self.index = {}
        self.docmap = {}
        self.term_freq = {}

    def __add_document(self, doc_id: str, text: str) -> None:
        stemmed_text = stem(stop(tokenize(text)))
        for token in stemmed_text:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            if doc_id not in self.term_freq:
                self.term_freq[doc_id] = Counter()
            if token not in self.term_freq[doc_id]:
                self.term_freq[doc_id][token] = 1
            else:
                self.term_freq[doc_id][token] += 1

    def get_documents(self, term: str) -> list[str]:
        return sorted(self.index.get(term.lower(), set()), key=lambda x: self.docmap[x]["id"])

    def build(self):
        with open("./data/movies.json", "r", encoding="utf-8") as f:
            movies: list[Movie] = cast(list[Movie], json.load(f)["movies"])
            for movie in movies:
                self.__add_document(str(movie["id"]), movie["title"] + " " + movie["description"])
                self.docmap[str(movie["id"])] = movie
            f.close()

    def save(self):
        if not os.path.exists("./cache"):
            os.makedirs("./cache")

        with open("./cache/index.pkl", "wb") as f:
            dump(self.index, f)
        f.close()
        with open("./cache/docmap.pkl", "wb") as f:
            dump(self.docmap, f)
        f.close()
        with open("./cache/term_freq.pkl", "wb") as f:
            dump(self.term_freq, f)
        f.close()

    def load(self):
        if not os.path.exists("./cache/index.pkl"):
            raise FileNotFoundError("Index file not found")
        if not os.path.exists("./cache/docmap.pkl"):
            raise FileNotFoundError("Docmap file not found")
        if not os.path.exists("./cache/term_freq.pkl"):
            raise FileNotFoundError("Term frequency file not found")
        with open("./cache/index.pkl", "rb") as f:
            self.index = load(f)
        f.close()
        with open("./cache/docmap.pkl", "rb") as f:
            self.docmap = load(f)
        f.close()
        with open("./cache/term_freq.pkl", "rb") as f:
            self.term_freq = load(f)
        f.close()

    def get_tf(self, doc_id: str, term: str) -> int:
        stemmed_term = stem(stop(tokenize(term)))
        if len(stemmed_term) != 1:
            raise ValueError("Can only get Token Frequency for a single token")

        return self.term_freq[doc_id][stemmed_term[0]]
