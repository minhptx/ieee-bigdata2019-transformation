import collections
import time
from pathlib import Path
from typing import List, Dict, Tuple

from datafc.syntactic.pattern import PatternTree
from datafc.transform.pattern_mapping.model import PatternMappingModel
from datafc.transform.validation import Validator


class Evaluator:
    def __init__(self):
        self.name_to_true_count: Dict[str, int] = collections.defaultdict(lambda: 0)
        self.name_to_total_count: Dict[str, int] = collections.defaultdict(lambda: 0)
        self.name_to_validation_count: Dict[str, int] = collections.defaultdict(lambda: 0)
        self.name_to_tp_validation_count: Dict[str, int] = collections.defaultdict(lambda: 0)
        self.name_to_fp_validation_count: Dict[str, int] = collections.defaultdict(lambda: 0)
        self.name_to_fn_validation_count: Dict[str, int] = collections.defaultdict(lambda: 0)
        self.name_to_time: Dict[str, float] = collections.defaultdict(lambda: 0.0)
        self.name_to_accuracy: Dict[str, float] = collections.defaultdict(lambda: 0.0)
        self.validator = Validator(0.1)

    def macro_mean_accuracy(self) -> float:
        return sum(self.name_to_accuracy.values()) * 1.0 / len(self.name_to_accuracy)

    def micro_mean_accuracy(self) -> float:
        return sum(self.name_to_true_count.values()) * 1.0 / sum(self.name_to_total_count.values())

    def validation_accuracy(self) -> float:
        return sum(self.name_to_validation_count.values()) * 1.0 / len(self.name_to_total_count)

    def validation_precision(self) -> float:
        sum_tp = sum(self.name_to_tp_validation_count.values())
        return sum_tp * 1.0 / (sum_tp + sum(self.name_to_fp_validation_count.values()))

    def validation_recall(self) -> float:
        sum_tp = sum(self.name_to_tp_validation_count.values())
        return sum_tp * 1.0 / (sum_tp + sum(self.name_to_fn_validation_count.values()))

    def average_time(self) -> float:
        return sum(self.name_to_time.values()) * 1.0 / len(self.name_to_time)

    def calculate_tree_result(self, name: str, original_values: List[str], transformed_values: List[str],
                              groundtruth_values: List[str]) -> Tuple[float, float, float]:
        saved_time = time.time()
        original_to_groundtruth: Dict[str, str] = {}

        for index, value in enumerate(original_values):
            # print(index, len(groundtruth_values))
            original_to_groundtruth[value] = groundtruth_values[index]

        transformed_tree = PatternTree.build_from_strings(transformed_values)
        print("Built from transformed values")
        original_tree = PatternTree.build_from_strings(original_values)
        print("Built from original values")

        transformation_model = PatternMappingModel(original_tree, transformed_tree, transformed_values)
        original_to_target_strings, scores = transformation_model.learn()

        print(original_to_target_strings, scores)

        Path("debug").mkdir(parents=True, exist_ok=True)

        print(len(original_values), len(original_to_target_strings))
        # print([x for x in original_values if x not in original_to_target_strings.keys()])
        assert len(original_values) == len(original_to_target_strings), \
            f"Dataset sizes should be the same ({len(original_values)} vs {len(original_to_target_strings)}) "

        validation_result = self.validator.validate([x[1] for x in original_to_target_strings], transformed_values,
                                                    scores)

        with (Path("debug") / f"{name}.txt").open("w") as writer:
            writer.write("Original,Transformed,Groundtruth,Result\n")

            print("Original to groundtruth", original_to_groundtruth)
            print("Original to transformed", original_to_target_strings)

            for original_value, transformed_value in original_to_target_strings:
                self.name_to_total_count[name] += 1
                result = False

                if transformed_value == original_to_groundtruth[original_value]:
                    result = True
                    self.name_to_true_count[name] += 1

                writer.write(f"{original_value},{transformed_value},"
                             f"{original_to_groundtruth[original_value]},{result}\n")

        print(name, self.name_to_total_count[name], self.name_to_true_count[name])
        self.name_to_accuracy[name] = self.name_to_true_count[name] * 1.0 / self.name_to_total_count[name]
        if validation_result and self.name_to_accuracy[name] > 0.95:
            self.name_to_validation_count[name] += 1
            self.name_to_tp_validation_count[name] += 1
        elif self.name_to_accuracy[name] > 0.95:
            self.name_to_fn_validation_count[name] += 1
        elif validation_result:
            self.name_to_fp_validation_count[name] += 1
        else:
            self.name_to_validation_count[name] += 1

        self.name_to_time[name] = time.time() - saved_time

        return self.name_to_true_count[name], self.name_to_total_count[name], self.name_to_accuracy[name]
