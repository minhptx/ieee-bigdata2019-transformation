import logging
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from datafc.syntactic.pattern import Pattern
from datafc.syntactic.token import TokenData
from datafc.transform.operators import Operation, Constant

logger = logging.getLogger(__name__)


class TokenMappingBaseModel(metaclass=ABCMeta):
    @abstractmethod
    def train_scoring_model(self, example_patterns: List[Pattern]):
        pass

    @staticmethod
    def find_best_mapping(token_pair_to_score, token_pair_to_operation) -> Tuple[float, Dict[int, Operation]]:

        index_to_operation = {}
        constant_indices = []
        post_delete_mapping = {}

        for idx1 in range(token_pair_to_operation.shape[0]):
            for idx2 in range(token_pair_to_operation.shape[1]):
                if isinstance(token_pair_to_operation[idx1][idx2], Constant):
                    index_to_operation[idx1] = token_pair_to_operation[idx1][idx2]
                    constant_indices.append(idx1)
                    break
            else:
                post_delete_mapping[len(post_delete_mapping)] = idx1

        token_pair_to_score = np.delete(token_pair_to_score, constant_indices, axis=0)
        score_matrix = np.array(token_pair_to_score)

        row_indices, col_indices = linear_sum_assignment(-score_matrix)

        for target_idx, original_idx in zip(row_indices, col_indices):
            pre_delete_index = post_delete_mapping[target_idx]
            index_to_operation[pre_delete_index] = token_pair_to_operation[pre_delete_index][original_idx]

        return score_matrix[row_indices, col_indices].sum() / len(row_indices), index_to_operation

    @abstractmethod
    def score_operation(self, operation: Operation) -> float:
        pass

    @abstractmethod
    def generate_candidates(self, source_token: TokenData, target_token: TokenData):
        pass

    def learn(self, original_pattern: Pattern, target_pattern: Pattern) -> \
            Tuple[List[Tuple[str, str]], float]:

        assert original_pattern.tokens, "Original pattern cannot be empty"
        assert target_pattern.tokens, "Target pattern cannot be empty"

        token_pair_to_score = np.zeros((len(target_pattern.tokens), len(original_pattern.tokens)))
        token_pair_to_operation = np.empty((len(target_pattern.tokens), len(original_pattern.tokens)), dtype=object)

        for idx1, target_token in enumerate(target_pattern.tokens):
            for idx2, original_token in enumerate(original_pattern.tokens):
                operation, score = self.generate_candidates(
                    original_token, target_token)

                logger.error(f"{operation} {score}")

                token_pair_to_operation[idx1][idx2] = operation
                token_pair_to_score[idx1][idx2] = score

        transformation_score, token_to_operation = \
            TokenMappingBaseModel.find_best_mapping(token_pair_to_score, token_pair_to_operation)

        if None in token_to_operation.values():
            return [(original_value, "") for original_value in original_pattern.values], -1.0

        try:
            assert len(target_pattern.tokens) == len(token_to_operation), "Cannot find satisfied mapping"
        except AssertionError:
            return [(original_val, "") for original_val in original_pattern.values], 0.0

        token_transformed_strings = []
        for idx in range(len(target_pattern.tokens)):
            token_transformed_strings.append(token_to_operation[idx].transform())

        original_to_target_string_pairs = []

        for idx, target_string in enumerate(original_pattern.values):
            target_string = "".join([transformed_strings[idx] for transformed_strings in token_transformed_strings])
            original_to_target_string_pairs.append((original_pattern.values[idx], target_string))

        for idx, operation in token_to_operation.items():
            print(operation.original_token, operation.target_token)
            print(idx, operation, operation.original_token.values[:3], operation.target_token.values[:3],
                  self.score_operation(operation))
        print("Original to target", original_to_target_string_pairs[:3])
        print("Transformation score", transformation_score)
        # print("Transformation score", transformation_score / math.sqrt(len(target_pattern.tokens)))
        print("--------------------------------------------------------------------")

        return original_to_target_string_pairs, transformation_score
