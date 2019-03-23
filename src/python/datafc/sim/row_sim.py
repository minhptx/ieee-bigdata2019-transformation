from datafc.data.row.source import Row
import numpy as np

from datafc.utils import create_ngrams
from datafc.utils import jaccard, to_number


def ngram_jaccard_row_attr(row1: Row, row2: Row, attribute_name: str):
    n_grams1 = np.array(create_ngrams(row1[attribute_name], 3))
    n_grams2 = np.array(create_ngrams(row2[attribute_name], 3))
    return jaccard(n_grams1, n_grams2)


def token_jaccard_row_attr(row1: Row, row2: Row, attribute_name: str):
    col1_array = np.array(row1[attribute_name].split())
    col2_array = np.array(row2[attribute_name].split())
    return jaccard(col1_array, col2_array)


def numeric_ratio_row_attr(row1: Row, row2: Row, attribute_name: str):
    num1 = to_number(row1[attribute_name])
    num2 = to_number(row2[attribute_name])
    if num1 and num2:
        return num1 * 1.0 / num2
    return 0.0


def numeric_exact_match_row_attr(row1: Row, row2: Row, attribute_name: str):
    num1 = to_number(row1[attribute_name])
    num2 = to_number(row2[attribute_name])

    if num1 is not None and num1 == num2:
        return 1.0
    return 0.0


def ngram_jaccard_row(row1: Row, row2: Row):
    text1 = row1.full_text()
    text2 = row2.full_text()
    n_grams1 = np.array(create_ngrams(text1, 3))
    n_grams2 = np.array(create_ngrams(text2, 3))
    return jaccard(n_grams1, n_grams2)


def token_jaccard_row(row1: Row, row2: Row):
    text1 = row1.full_text()
    text2 = row2.full_text()
    tokens1 = np.array(text1.split())
    tokens2 = np.array(text2.split())
    return jaccard(tokens1, tokens2)


