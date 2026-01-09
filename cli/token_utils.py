import string
from typing import TypedDict

from nltk.stem import PorterStemmer


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


class Movie(TypedDict):
    id: int
    title: str
    description: str
