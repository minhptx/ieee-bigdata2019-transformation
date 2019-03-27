import collections
import logging
from typing import Dict, Tuple, List

from datafc.repr.column import Column
from datafc.sim.column_sim import syntactic_sim
from datafc.syntactic.pattern import PatternTree, PatternNode
from datafc.transform.token_mapping.base import TokenMappingBaseModel
from datafc.transform.token_mapping.token_attr import TokenAttrMappingModel

logger = logging.getLogger(__name__)


class PatternMappingModel:
    def __init__(self, original_tree: PatternTree, target_tree: PatternTree, transformed_values):
        self.original_tree = original_tree
        self.target_tree = target_tree
        self.transformed_values = transformed_values

        self.token_mapper: TokenMappingBaseModel = TokenAttrMappingModel()
        self.token_mapper.train_scoring_model(
            original_tree.get_patterns_by_layers([2, 3, 4]) + target_tree.get_patterns_by_layers([2, 3, 4]))
        self.node_to_str_pair_score: Dict[PatternNode, Tuple[List[Tuple[str, str]], float]] = collections.defaultdict(
            list)

    def learn(self):
        self.node_to_str_pair_score: Dict[PatternNode, Tuple[List[Tuple[str, str]], float]] = collections.defaultdict(
            list)

        layer = 4
        layers = [4]
        while abs(len(self.original_tree.node_in_layers[layer]) - len(self.target_tree.node_in_layers[layer])) == abs(
                len(self.original_tree.node_in_layers[layer - 1]) - len(self.target_tree.node_in_layers[layer - 1])) \
                and len(self.target_tree.node_in_layers[layer - 1]) == len(self.target_tree.node_in_layers[layer]):
            layer = layer - 1
            layers.append(layer)
            if layer == 0:
                break

        target_col = Column("Target", "Target", self.transformed_values)
        best_score = 0

        for layer in [4, 3, 2, 1]:
            transformed_col = Column("Transformed", "Transformed")
            logger.debug("Diff", layer, len(self.original_tree.node_in_layers[layer]), abs(
                len(self.original_tree.node_in_layers[layer]) - len(self.target_tree.node_in_layers[layer])))
            original_nodes = sorted(self.original_tree.node_in_layers[layer],
                                    key=lambda x: len(x.value.tokens), reverse=True)
            transformed_nodes = sorted(self.target_tree.node_in_layers[layer],
                                       key=lambda x: len(x.value.tokens), reverse=True)

            # self.node_pair_to_score: Dict[PatternNode, Dict[PatternNode, float]] = {}
            node_to_str_pair_score = {}

            for original_node in original_nodes:
                temp_result = []
                for target_node in transformed_nodes:
                    result = self.token_mapper.learn(original_node.value,
                                                     target_node.value)

                    temp_result.append(result)

                node_to_str_pair_score[original_node] = max(temp_result, key=lambda x: x[1])

                value_tuples = list(zip(*node_to_str_pair_score[original_node][0]))
                transformed_col.extend_values(list(value_tuples[1]))

            pattern_score = syntactic_sim(target_col, transformed_col)

            if pattern_score > best_score:
                best_score = pattern_score
                self.node_to_str_pair_score = node_to_str_pair_score.copy()

        original_to_target_strings = []

        scores = []

        for node, (current_original_to_target_strings, score) in self.node_to_str_pair_score.items():
            original_to_target_strings.extend(current_original_to_target_strings)

            print("-----------------------------------------------------------------")
            print("Best transformation", score, node.value)
            print("Best transformation", current_original_to_target_strings[:3])

            scores.append(score)

        return original_to_target_strings, scores
