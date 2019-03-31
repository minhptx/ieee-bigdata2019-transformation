import logging
from typing import List

from datafc.syntactic.pattern import PatternTree
from datafc.transform.pattern_mapping.model import PatternMappingModel

logger = logging.getLogger(__name__)


class TransformationModel:
    def __init__(self):
        self.pattern_mapper = PatternMappingModel()

    @staticmethod
    def generate_patterns(original_values: List[str], transformed_values: List[str]):
        transformed_tree = PatternTree.build_from_strings(transformed_values)
        logger.debug("Built from transformed values")

        original_tree = PatternTree.build_from_strings(original_values)
        logger.debug("Built from original values")

        return original_tree, transformed_tree

    def learn_top_k(self, original_values: List[str], transformed_values: List[str], k: int, validated=True):
        original_tree, transformed_tree = self.generate_patterns(original_values, transformed_values)

        original_to_transformed_pairs_list, scores = self.pattern_mapper.learn_top_k(original_tree, transformed_tree, k)

        original_to_transformed_results = []

        for original_to_transformed_pairs in original_to_transformed_pairs_list:
            for idx, (original_value, transformed_value, valid_result) in enumerate(original_to_transformed_pairs):
                if len(original_to_transformed_results) <= idx:
                    original_to_transformed_results.append((original_value, [transformed_value], [valid_result]))
                else:
                    assert original_to_transformed_results[idx][0] == original_value, \
                        "Wrong order between top-k results"
                    original_to_transformed_results[idx][1].append(transformed_value)
                    original_to_transformed_results[idx][2].append(valid_result)

        return original_to_transformed_results, scores

    def learn(self, original_values: List[str], transformed_values: List[str], validated=True):
        original_tree, transformed_tree = self.generate_patterns(original_values, transformed_values)

        original_to_transformed_tuples, score = self.pattern_mapper.learn(original_tree, transformed_tree)

        return original_to_transformed_tuples, score
