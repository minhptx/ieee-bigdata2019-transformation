import logging
from dataclasses import dataclass
from typing import Dict, Tuple, List

from datafc.repr.column import Column
from datafc.syntactic.pattern import PatternTree, PatternNode
from datafc.transform.token_mapping.token_sim import TokenSimMappingModel
from datafc.transform.validation import Validator

logger = logging.getLogger(__name__)


@dataclass
class TransformedResult:
    original_target_pairs: List[Tuple[str, str]]
    score: float


class PatternMappingModel:
    def __init__(self):
        self.token_mapper: TokenSimMappingModel = TokenSimMappingModel()
        self.validator: Validator = Validator(0.1)

    def learn(self, original_tree: PatternTree, target_tree: PatternTree):

        self.token_mapper.train_scoring_model(original_tree.get_patterns_by_layers([1, 2, 3, 4], in_groups=True) +
                                              target_tree.get_patterns_by_layers([1, 2, 3, 4], in_groups=True))

        final_node_to_result: Dict[PatternNode, TransformedResult] = {}

        target_col = Column(values=target_tree.values)
        best_score, best_level = 0, 0

        for layer in [4, 3, 2, 1]:

            transformed_col = Column()

            original_nodes = sorted(original_tree.node_in_layers[layer],
                                    key=lambda x: len(x.value.tokens), reverse=True)
            transformed_nodes = sorted(target_tree.node_in_layers[layer],
                                       key=lambda x: len(x.value.tokens), reverse=True)

            node_to_result: Dict[PatternNode, TransformedResult] = {}

            for original_node in original_nodes:
                transformed_results = []
                for target_node in transformed_nodes:
                    result = self.token_mapper.learn(original_node.value,
                                                     target_node.value)

                    transformed_results.append(TransformedResult(original_target_pairs=result[0], score=result[1]))

                node_to_result[original_node] = max(transformed_results, key=lambda x: x.score)

                value_tuples = list(zip(*node_to_result[original_node].original_target_pairs))
                transformed_col.extend_values(list(value_tuples[1]))

            pattern_score = self.token_mapper.scoring_model.predict_similarity(target_col, transformed_col)

            if pattern_score > best_score:
                best_score = pattern_score
                best_level = layer
                final_node_to_result = node_to_result.copy()

        validated_original_to_transformed_tuples = []

        scores = []

        for node, result in final_node_to_result.items():

            for original_value, transformed_value in result.original_target_pairs:
                validation_result = self.validator.validate_result(transformed_value, original_tree, target_tree,
                                                                   result.score, best_level)
                validated_original_to_transformed_tuples.append((original_value, transformed_value, validation_result))

            logger.debug("Best transformation %f %s %s" %
                         (result.score, node.value, validated_original_to_transformed_tuples[-3:]))
            logger.debug("-----------------------------------------------------------------")

            scores.append(result.score)

        return validated_original_to_transformed_tuples, sum(scores) * 1.0 / len(scores)

    def learn_top_k(self, original_tree: PatternTree, target_tree: PatternTree, k: int):
        self.token_mapper.train_scoring_model(original_tree.get_patterns_by_layers([1, 2, 3, 4], in_groups=True) +
                                              target_tree.get_patterns_by_layers([1, 2, 3, 4], in_groups=True))

        final_node_to_results: Dict[PatternNode, List[TransformedResult]] = {}
        target_col = Column(values=target_tree.values)
        best_score, best_level = 0, 0

        for layer in [4, 3, 2]:

            transformed_col = Column()

            original_nodes = sorted(original_tree.node_in_layers[layer],
                                    key=lambda x: len(x.value.tokens), reverse=True)
            transformed_nodes = sorted(target_tree.node_in_layers[layer],
                                       key=lambda x: len(x.value.tokens), reverse=True)

            # self.node_pair_to_score: Dict[PatternNode, Dict[PatternNode, float]] = {}
            node_to_results = {}

            for original_node in original_nodes:
                token_mapping_results = []
                for target_node in transformed_nodes:
                    validated_original_to_transformed_tuples_list, scores_list = self.token_mapper.learn_top_k(
                        original_node.value, target_node.value, k)

                    token_mapping_results.append([TransformedResult(*result) for
                                                  result in
                                                  zip(validated_original_to_transformed_tuples_list, scores_list)])

                node_to_results[original_node] = max(token_mapping_results, key=lambda x: x[0].score)

                value_tuples = list(zip(*node_to_results[original_node][0].original_target_pairs))
                transformed_col.extend_values(list(value_tuples[1]))

            pattern_score = self.token_mapper.scoring_model.predict_similarity(target_col, transformed_col)

            if pattern_score > best_score:
                best_score = pattern_score
                best_level = layer
                final_node_to_results = node_to_results.copy()

        validated_original_to_transformed_tuples_list = [[] for _ in range(k)]

        scores_list = [[] for _ in range(k)]

        for node, results in final_node_to_results.items():
            for idx, result in enumerate(results):
                for original_value, transformed_value in result.original_target_pairs:
                    validation_result = self.validator.validate_result(transformed_value, original_tree, target_tree,
                                                                       result.score, best_level)
                    validated_original_to_transformed_tuples_list[idx].append(
                        (original_value, transformed_value, validation_result))

                logger.debug("Best transformation %f %s %s" %
                             (result.score, node.value, validated_original_to_transformed_tuples_list[idx][-3:]))
                logger.debug("-----------------------------------------------------------------")

                scores_list[idx].append(result.score)
        return validated_original_to_transformed_tuples_list, [sum(scores) * 1.0 / len(scores) for scores in
                                                               scores_list]
