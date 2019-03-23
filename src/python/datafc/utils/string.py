import locale
from typing import Optional

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def jaccard(x, y):
    try:
        return np.intersect1d(x, y).size / np.union1d(x, y).size
    except Exception:
        return 0


def cosine(x: str, y: str):
    vectorizer = CountVectorizer()
    try:
        vectors = vectorizer.fit_transform([x, y]).toarray()
    except Exception as _:
        return 0
    return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]


def to_number(num_str: str) -> Optional[float]:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    try:
        return locale.atof(num_str)
    except ValueError:
        return None
