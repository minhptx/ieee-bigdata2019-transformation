from typing import List, Tuple

from datafc.syn.pattern import PatternTree


class Validator:
    def __init__(self, threshold: float):
        self.threshold: float = threshold

    def validate_results(
        self,
        original_transformed_pairs: List[Tuple[str, str]],
        original_tree: PatternTree,
        target_tree: PatternTree,
        level: int,
    ):
        transformed_values = [x[1] for x in original_transformed_pairs]
        transformed_tree = PatternTree.build_from_strings(transformed_values)

        is_syntactic = set(transformed_tree.get_patterns_by_layers([level])).issubset(
            set(target_tree.get_patterns_by_layers([level]))
        )

    def validate_result(
        self,
        original_transformed_pair: Tuple[str, str],
        target_tree: PatternTree,
        score: float,
        next_score: float,
        level: int,
    ):
        pattern_tree = PatternTree.build_from_strings([original_transformed_pair[1]])

        is_syntactic = set(pattern_tree.get_patterns_by_layers([level])).issubset(
            set(target_tree.get_patterns_by_layers([level]))
        )

        if is_syntactic and (score - next_score) > self.threshold:
            return False, is_syntactic, round(score, 2), round(next_score, 2)
        return True, is_syntactic, round(score, 2), round(next_score, 2)
