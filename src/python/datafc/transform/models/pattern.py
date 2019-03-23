import collections
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import KFold

from datafc.sim.column_sim import values_jaccard, ngram_jaccard, syntactic_sim
from datafc.repr.column import Column
from datafc.ml.classification.multi_binary import MultiBinary
from datafc.syntactic.pattern import PatternTree, PatternNode, Pattern
from datafc.transform.operators import Operation, Constant


class PatternTransformationProgram:
    def __init__(self, scoring_model: MultiBinary):
        self.scoring_model: MultiBinary = scoring_model

    @staticmethod
    def best_mapping(token_pair_to_score, token_pair_to_operation) -> Tuple[float, Dict[int, Operation]]:
        # indices = np.argmax(token_pair_to_score, axis=1)
        # return token_pair_to_score[range(len(indices)), indices].sum(), \
        #        {idx1: idx2 for idx1, idx2 in zip(range(len(indices)), indices)}
        index_to_operation = {}
        constant_indices = []
        post_delete_mapping = {}

        idx = 0
        for idx1 in range(token_pair_to_operation.shape[0]):
            for idx2 in range(token_pair_to_operation.shape[1]):
                if isinstance(token_pair_to_operation[idx1][idx2], Constant):
                    print("Constant: ", idx1, token_pair_to_operation[idx1][idx2])
                    index_to_operation[idx1] = token_pair_to_operation[idx1][idx2]
                    constant_indices.append(idx1)
                    break
            else:
                post_delete_mapping[idx] = idx1
                idx += 1

        token_pair_to_score = np.delete(token_pair_to_score, constant_indices, axis=0)

        print("Post mapping: ", post_delete_mapping)

        score_matrix = np.array(token_pair_to_score)

        print("Score matrix: ", score_matrix)

        row_indices, col_indices = linear_sum_assignment(-score_matrix)

        print("Mapping: ", list(zip(row_indices, col_indices)))

        for target_idx, original_idx in zip(row_indices, col_indices):
            pre_delete_index = post_delete_mapping[target_idx]
            index_to_operation[pre_delete_index] = token_pair_to_operation[pre_delete_index][original_idx]

        return score_matrix[row_indices, col_indices].sum() / len(row_indices), index_to_operation

    def learn_transformation(self, original_pattern: Pattern,
                             target_pattern: Pattern, transformed_values: List[str]) -> \
            Tuple[List[Tuple[str, str]], float]:
        print("Original pattern", original_pattern)
        print("Original pattern values", [x.values[:3] for x in original_pattern.tokens])
        print("Target pattern", target_pattern)
        print("Target pattern values", [x.values[:3] for x in target_pattern.tokens])

        assert original_pattern.tokens, "Original pattern cannot be empty"
        assert target_pattern.tokens, "Target pattern cannot be empty"
        token_pair_to_score = np.zeros((len(target_pattern.tokens), len(original_pattern.tokens)))
        token_pair_to_operation = np.empty((len(target_pattern.tokens), len(original_pattern.tokens)), dtype=object)
        for idx1, target_token in enumerate(target_pattern.tokens):
            # print("Target values", target_token.values)
            for idx2, original_token in enumerate(original_pattern.tokens):
                # print("Original values", original_token.values)
                operation, score = Operation.find_top_transformations(
                    original_token, target_token, self.scoring_model)
                # print(original_token, target_token, score, operation)
                token_pair_to_operation[idx1][idx2] = operation
                if isinstance(operation, Constant):
                    score = score * len(target_token.values) * 1.0 / len(transformed_values)
                token_pair_to_score[idx1][idx2] = score

        transformation_score, token_to_operation = \
            PatternTransformationProgram.best_mapping(token_pair_to_score, token_pair_to_operation)

        transformed_strings = [""] * len(original_pattern.tokens[0].values)

        if None in token_to_operation.values():
            return [(original_value, "") for original_value in original_pattern.values], -1.0

        # print(token_to_operation)
        try:
            assert len(target_pattern.tokens) == len(token_to_operation), "Number of tokens in target pattern is higher"
        except AssertionError:
            return [(original_val, "") for original_val in original_pattern.values], 0.0
        for idx in range(len(target_pattern.tokens)):
            transformed_token_strings = token_to_operation[idx].transform()
            # print(token_to_operation_score[idx][0].original_values)

            assert len(transformed_strings) == len(transformed_token_strings), \
                "Different number of strings after transformation"
            for idx1 in range(len(transformed_strings)):
                transformed_strings[idx1] += transformed_token_strings[idx1]

        original_to_target_string_pairs = []

        for idx, target_string in enumerate(transformed_strings):
            original_to_target_string_pairs.append((original_pattern.values[idx], target_string))

        print("Transformation Plan: ")
        for idx, operation in token_to_operation.items():
            print(operation.original_token, operation.target_token)
            print(idx, operation, operation.original_token.values[:3], operation.target_token.values[:3],
                  operation.score(self.scoring_model))
        print("Original to target", original_to_target_string_pairs[:3])
        print("Transformation score", transformation_score)
        # print("Transformation score", transformation_score / math.sqrt(len(target_pattern.tokens)))
        print("--------------------------------------------------------------------")
        return original_to_target_string_pairs, transformation_score


class PatternTransformationModel:
    def __init__(self, original_tree: PatternTree, target_tree: PatternTree, transformed_values):
        self.original_tree = original_tree
        self.target_tree = target_tree
        self.transformed_values = transformed_values
        self.scoring_model: MultiBinary = MultiBinary[Column](PatternTransformationModel.compute_sim)

        print("Prepare for training")
        self.train_scoring_model()
        print("Finish training")

        self.transform_program: PatternTransformationProgram = PatternTransformationProgram(self.scoring_model)
        self.node_to_str_pair_score: Dict[PatternNode, Tuple[List[Tuple[str, str]], float]] = collections.defaultdict(
            list)

    @staticmethod
    def compute_sim(col1: Column, col2: Column) -> List[float]:
        return [values_jaccard(col1, col2), ngram_jaccard(col1, col2), syntactic_sim(col1, col2)]

    def train_scoring_model(self):
        # if (Path("model") / "model.pkl").exists():
        #     self.scoring_model.load(Path("model") / "model.pkl")
        idx = 0
        label_cols: List[Tuple[str, Column]] = []
        patterns = self.original_tree.get_patterns_by_layers([2, 3, 4]) + \
                   self.target_tree.get_patterns_by_layers([2, 3, 4])
        print("Number of patterns: ", len(patterns))
        for idx1, pattern_node in enumerate(patterns):
            pattern = pattern_node.value
            for token in pattern.tokens:
                print("Num values: ", len(token.values), len(pattern.values))
                if len(token.values) <= 4:
                    continue
                idx += 1
                num_splits = 2
                # if len(token.values) <= 2:
                #     num_splits = len(token.values)
                kf = KFold(n_splits=num_splits)
                for train_indices, test_indices in kf.split(token.values):
                    train_values = []
                    for index in train_indices:
                        train_values.append(token.values[index])
                    column = Column(str(idx), str(idx), train_values)
                    label_cols.append((str(idx), column))
        # print("Len labeled cols", len(label_cols))
        try:
            self.scoring_model.train(label_cols)
        except Exception as e:
            print(e)
            print("No training data")
            self.scoring_model.model.coef_ = [0.5, 0.5]
        print(self.scoring_model.model.coef_)
        # self.scoring_model.save(Path("model") / "model.pkl")

    def best_transformation_in_branch(self, node: PatternNode) -> Dict[PatternNode,
                                                                       Tuple[List[Tuple[str, str]], float]]:
        current_string_pairs, current_node_score = self.node_to_str_pair_score[node]
        if node.value.level != 2 and node.children:
            current_node_to_str_pair_score = {}
            scores = []
            for child in node.children:
                child_node_to_op_score = self.best_transformation_in_branch(child)
                for node, (string_pairs, score) in child_node_to_op_score.items():
                    current_node_to_str_pair_score[node] = (string_pairs, score)
                    scores.append(score)
            if sum(scores) / len(scores) > current_node_score:
                return current_node_to_str_pair_score
        return {node: (current_string_pairs, current_node_score)}

    def learn_and_transform_between_trees(self):
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
            print("Diff", layer, len(self.original_tree.node_in_layers[layer]), abs(
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
                    result = self.transform_program.learn_transformation(original_node.value,
                                                                         target_node.value, self.transformed_values)
                    temp_result.append(result)

                node_to_str_pair_score[original_node] = max(temp_result, key=lambda x: x[1])

                value_tuples = list(zip(*node_to_str_pair_score[original_node][0]))
                transformed_col.extend_values(list(value_tuples[1]))

            pattern_score = syntactic_sim(target_col, transformed_col)
            print(self.scoring_model.model.coef_)
            print("Pattern score", pattern_score)

            if pattern_score > best_score:
                best_score = pattern_score
                self.node_to_str_pair_score = node_to_str_pair_score.copy()

        original_to_target_strings = []
        # for original_node in self.original_tree.node_in_layers[layer]:
        # node_to_str_pair_score = self.best_transformation_in_branch(original_node)

        scores = []

        for node, (current_original_to_target_strings, score) in self.node_to_str_pair_score.items():
            original_to_target_strings.extend(current_original_to_target_strings)

            print("-----------------------------------------------------------------")
            print("Best transformation", score, node.value)
            print("Best transformation", current_original_to_target_strings[:3])
            scores.append(score)

        return original_to_target_strings, scores
