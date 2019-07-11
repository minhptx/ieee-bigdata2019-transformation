from datafc.syntactic.pattern import PatternTree


class Validator:
    def __init__(self, threshold: float):
        self.threshold: float = threshold

    def validate_result(
        self, str_value, original_tree: PatternTree, target_tree: PatternTree, score: float, level: int
    ):
        pattern_tree = PatternTree.build_from_strings([str_value])

        is_syntactic = set(pattern_tree.get_patterns_by_layers([level])).issubset(set(target_tree.get_patterns_by_layers([level])))

        if (
            is_syntactic
            and score > self.threshold
        ):
            return False
        return True
