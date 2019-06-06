import locale
import logging
from typing import Optional

import numpy as np
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


def jaccard(x, y):
    if x and y:
        return np.intersect1d(x, y).size / np.union1d(x, y).size
    return 0


def cosine(x: str, y: str):
    vectorizer = CountVectorizer()
    try:
        vectors = vectorizer.fit_transform([x, y]).toarray()
    except Exception as e:
        logger.error(e)
        return 0
    return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]


def to_number(num_str: str) -> Optional[float]:
    locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
    try:
        return locale.atof(num_str)
    except ValueError:
        return None


class Spacy:
    model = spacy.load("en_core_web_lg")
