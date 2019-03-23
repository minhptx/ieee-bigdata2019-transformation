import numpy as np
from scipy.stats import ks_2samp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity

from datafc.repr.column import Column
from datafc.repr.helpers import create_ngrams
from datafc.syntactic.token import TokenData
from datafc.utils.string import jaccard


def name_jaccard(col1: Column, col2: Column) -> float:
    n_grams1 = np.array(create_ngrams(col1.name, 2))
    n_grams2 = np.array(create_ngrams(col2.name, 2))
    return jaccard(n_grams1, n_grams2)


def text_cosine(col1: Column, col2: Column) -> float:
    count_vectorizer = CountVectorizer()
    word_counts = count_vectorizer.fit_transform([col1.text(), col2.text()])

    return cosine_similarity(word_counts[:1], word_counts[1:])[0][0]


def text_jaccard(col1: Column, col2: Column) -> float:
    col1_array = np.array(col1.text().split(" "))
    col2_array = np.array(col2.text().split(" "))

    return jaccard(col1_array, col2_array)


def ngram_jaccard(col1: Column, col2: Column) -> float:
    n_grams1 = [x for value in col1.values for x in create_ngrams(value, 2)]
    n_grams2 = [x for value in col2.values for x in create_ngrams(value, 2)]

    return jaccard(n_grams1, n_grams2)


def values_jaccard(col1: Column, col2: Column) -> float:
    col1_array = np.array(col1.values)
    col2_array = np.array(col2.values)

    return jaccard(col1_array, col2_array)


def numeric_ks(col1: Column, col2: Column) -> float:
    if not col1.numeric_values or not col2.numeric_values:
        return 0
    return ks_2samp(col1.numeric_values, col2.numeric_values)[1]


def histogram_nmi(col1: Column, col2: Column) -> float:
    word_to_idx = {word: idx for idx, word in enumerate(list(set(col1.values + col2.values)))}
    indices1 = [word_to_idx[value] for value in col1.values]
    indices2 = [word_to_idx[value] for value in col2.values]

    return normalized_mutual_info_score(indices1, indices2)


def length_diff(col1: Column, col2: Column) -> float:
    sum_length1 = sum([len(x) for x in col1.values]) * 1.0 / len(col1.values)
    sum_length2 = sum([len(x) for x in col2.values]) * 1.0 / len(col2.values)
    if max(sum_length1, sum_length2) == 0:
        return 0
    return abs(sum_length1 - sum_length2) * 1.0 / max(sum_length1, sum_length2)


def syntactic_sim(col1: Column, col2: Column) -> float:
    token_lists1 = []
    for str_value in col1.values:
        token_lists1.append(" ".join([x.name for x in TokenData.get_basic_tokens(str_value)]))

    token_lists2 = []
    for str_value in col2.values:
        token_lists2.append(" ".join([x.name for x in TokenData.get_basic_tokens(str_value)]))

    return jaccard(token_lists1, token_lists2)


def syntactic_ratio(col1: Column, col2: Column) -> float:
    tokens1: set = set(TokenData.get_basic_tokens(col1.values[0]))
    tokens2: set = set(TokenData.get_basic_tokens((col2.values[0])))

    for str_value in col1.values[1:]:
        tokens1 = set(tokens1).intersection(set(TokenData.get_basic_tokens(str_value)))

    for str_value in col2.values[1:]:
        tokens2 = set(tokens2).intersection(set(TokenData.get_basic_tokens(str_value)))

    return len(tokens1.intersection(tokens2)) * 1.0 / len(tokens1.union(tokens2))


def word2vec_sim(col1: Column, col2: Column):
    col1_matrix = [x.split() for x in col1.values]
    col2_matrix = [x.split() for x in col2.values]

    # col1_w2v_matrix = word2vec.query(col1_matrix)
    # col2_w2v_matrix = word2vec.query(col2_matrix)
    return 0
