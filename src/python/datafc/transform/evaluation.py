import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd

from datafc.transform.model import TransformationModel

logger = logging.getLogger(__name__)


@dataclass
class TopKTransformationResult:
    correct_count: int = 0
    total_count: int = 0
    top_k_correct_count: int = 0
    mrr_count: float = 0
    running_time: float = 0
    validation_tp_count: int = 0
    validation_tn_count: int = 0
    validation_fp_count: int = 0
    validation_fn_count: int = 0
    is_correct_transformation: int = 1


class Evaluator:
    def __init__(self):
        self.name_to_result: Dict[str, TopKTransformationResult] = {}

    def micro_accuracy(self, name):
        return self.name_to_result[name].correct_count * 1.0 / self.name_to_result[name].total_count

    def micro_top_k_accuracy(self, name):
        return self.name_to_result[name].top_k_correct_count * 1.0 / self.name_to_result[name].total_count

    def mrr_score(self, name):
        return self.name_to_result[name].mrr_count / self.name_to_result[name].total_count

    def exact_accuracy(self):
        return sum([x.is_correct_transformation for x in self.name_to_result.values()]) / len(self.name_to_result)

    def micro_mean_accuracy(self):
        return sum([x.correct_count for x in self.name_to_result.values()]) * 1.0 / \
               sum([x.total_count for x in self.name_to_result.values()])

    def macro_mean_accuracy(self):
        return sum([self.micro_accuracy(name) for name in self.name_to_result]) / len(self.name_to_result)

    def micro_mean_top_k_accuracy(self):
        return sum([x.top_k_correct_count for x in self.name_to_result.values()]) * 1.0 / \
               sum([x.total_count for x in self.name_to_result.values()])

    def macro_mean_top_k_accuracy(self):
        return sum([self.micro_top_k_accuracy(name) for name in self.name_to_result]) / len(self.name_to_result)

    def micro_mrr_score(self):
        return sum([x.mrr_count for x in self.name_to_result.values()]) * 1.0 / \
               sum([x.total_count for x in self.name_to_result.values()])

    def macro_mrr_score(self):
        return sum([self.mrr_score(name) for name in self.name_to_result]) / len(self.name_to_result)

    def validation_precision(self, name):
        try:
            return self.name_to_result[name].validation_tp_count * 1.0 / \
                   (self.name_to_result[name].validation_tp_count + self.name_to_result[name].validation_fp_count)
        except ZeroDivisionError:
            return 0

    def validation_recall(self, name):
        try:
            return self.name_to_result[name].validation_tp_count * 1.0 / \
                   (self.name_to_result[name].validation_tp_count + self.name_to_result[name].validation_tn_count)
        except ZeroDivisionError:
            return 0

    def validation_f1(self, name):
        precision = self.validation_precision(name)
        recall = self.validation_recall(name)
        try:
            return (2 * precision * recall) / (precision + recall)
        except ZeroDivisionError:
            return 0

    def mean_validation_precision(self):
        return sum([x.validation_tp_count for x in self.name_to_result.values()]) * 1.0 / \
               sum([x.validation_tp_count + x.validation_fp_count for x in self.name_to_result.values()])

    def mean_validation_recall(self):
        return sum([x.validation_tp_count for x in self.name_to_result.values()]) * 1.0 / \
               sum([x.validation_tp_count + x.validation_tn_count for x in self.name_to_result.values()])

    def mean_validation_f1(self):
        precision = self.mean_validation_precision()
        recall = self.mean_validation_recall()

        return (2 * precision * recall) / (precision + recall)

    def mean_running_time(self):
        return sum([x.running_time for x in self.name_to_result.values()]) / len(self.name_to_result)

    def check_validation_result(self, name, validation_result, transformed_value, groundtruth_value):
        if transformed_value == groundtruth_value:
            if validation_result:
                self.name_to_result[name].validation_fp_count += 1
            else:
                self.name_to_result[name].validation_tn_count += 1
        else:
            if validation_result:
                self.name_to_result[name].validation_tp_count += 1
            else:
                self.name_to_result[name].validation_fn_count += 1

    def check_transformation_result(self, name: str, original_to_groundtruth_values: Dict[str, str],
                                    original_to_transformed_values: List[Tuple[str, List[str]]]):

        Path("debug").mkdir(parents=True, exist_ok=True)
        with (Path("debug") / f"{name}.txt").open("w") as writer:
            writer.write("Original,Transformed,Groundtruth,Result\n")

            logger.debug("Original to groundtruth %s" % original_to_groundtruth_values)
            logger.debug("Original to transformed %s" % original_to_transformed_values)

            for original_value, transformed_value, _ in original_to_transformed_values:
                self.name_to_result[name].total_count += 1
                result = False

                if transformed_value == original_to_groundtruth_values[original_value]:
                    result = True
                    self.name_to_result[name].correct_count += 1
                else:
                    self.name_to_result[name].is_correct_transformation = 0

                writer.write(f"{original_value},{transformed_value},"
                             f"{original_to_groundtruth_values[original_value]},{result}\n")

    def check_top_k_transformation_result(self, name: str, original_to_groundtruth_values: Dict[str, str],
                                          original_to_k_transformed_values: List[Tuple[str, List[str]]], k: int):
        Path("debug").mkdir(parents=True, exist_ok=True)
        with (Path("debug") / f"{name}_top_{k}.txt").open("w") as writer:
            writer.write("Original,Transformed,Groundtruth,Result\n")

            for original_value, transformed_values, validated_results in original_to_k_transformed_values:
                self.name_to_result[name].total_count += 1
                result = True

                self.check_validation_result(name, validated_results[0],
                                             transformed_values[0], original_to_groundtruth_values[original_value])

                if transformed_values[0] == original_to_groundtruth_values[original_value]:
                    self.name_to_result[name].correct_count += 1
                    self.name_to_result[name].top_k_correct_count += 1
                    self.name_to_result[name].mrr_count += 1
                else:
                    try:
                        correct_rank = transformed_values.index(original_to_groundtruth_values[original_value])
                    except ValueError:
                        correct_rank = -1
                    if correct_rank != -1:
                        self.name_to_result[name].mrr_count += (1.0 / (correct_rank + 1))
                        self.name_to_result[name].top_k_correct_count += 1
                    else:
                        result = False
                        self.name_to_result[name].is_correct_transformation = 0

                writer.write(f"{original_value},{transformed_values},"
                             f"{original_to_groundtruth_values[original_value]},{result}\n")

    def run_normal_experiment(self, name: str, original_values: List[str], target_values: List[str],
                              groundtruth_values: List[str]):

        self.name_to_result[name] = TopKTransformationResult()

        original_to_groundtruth_values: Dict[str, str] = {}

        for index, value in enumerate(original_values):
            original_to_groundtruth_values[value] = groundtruth_values[index]

        starting_time = time.time()

        transformation_model = TransformationModel()
        validated_original_to_transformed_values, scores = transformation_model.learn(original_values, target_values)

        logger.debug(validated_original_to_transformed_values)

        self.name_to_result[name].running_time = time.time() - starting_time

        assert len(original_values) == len(validated_original_to_transformed_values), \
            f"Dataset sizes before and after transformation " \
                f"should be the same ({len(original_values)} vs {len(validated_original_to_transformed_values)}) "

        self.check_transformation_result(name, original_to_groundtruth_values, validated_original_to_transformed_values)

    def run_top_k_experiment(self, name: str, original_values: List[str], target_values: List[str],
                             groundtruth_values: List[str], k: int):

        self.name_to_result[name] = TopKTransformationResult()

        original_to_groundtruth_values: Dict[str, str] = {}

        for index, value in enumerate(original_values):
            original_to_groundtruth_values[value] = groundtruth_values[index]

        starting_time = time.time()

        transformation_model = TransformationModel()
        validated_original_to_k_transformed_values, scores = transformation_model.learn_top_k(original_values,
                                                                                              target_values, k)

        logger.debug(validated_original_to_k_transformed_values)

        self.name_to_result[name].running_time = time.time() - starting_time

        assert len(original_values) == len(validated_original_to_k_transformed_values), \
            f"Dataset sizes before and after transformation " \
                f"should be the same ({len(original_values)} vs {len(validated_original_to_k_transformed_values)}) "

        self.check_top_k_transformation_result(name, original_to_groundtruth_values,
                                               validated_original_to_k_transformed_values, k)

    def report_file(self, name):
        logger.info(f"Transformation Accuracy: {self.micro_accuracy(name)} "
                    f"--- Top K Accuracy: {self.micro_top_k_accuracy(name)}"
                    f"--- MRR Score: {self.mrr_score(name)}"
                    f"--- Validation Precision: {self.validation_precision(name)} "
                    f"--- Validation Recall: {self.validation_recall(name)} "
                    f"--- Validation F1: {self.validation_f1(name)} "
                    f"--- Running Time: {self.name_to_result[name].running_time}")

    def report_dataset(self):
        logger.info(f"Macro Mean Accuracy: {self.macro_mean_accuracy()} "
                    f"--- Micro Mean Accuracy: {self.micro_mean_accuracy()} "
                    f"--- Macro Top-K Accuracy: {self.macro_mean_top_k_accuracy()} "
                    f"--- Micro Top-K Accuracy: {self.micro_mean_top_k_accuracy()} "
                    f"--- Macro MRR Score: {self.macro_mrr_score()} "
                    f"--- Micro MRR Score: {self.micro_mrr_score()} "
                    f"--- Exact Accuracy: {self.exact_accuracy()} "
                    f"--- Validation Precision: {self.mean_validation_precision()} "
                    f"--- Validation Recall: {self.mean_validation_recall()} "
                    f"--- Validation F-measure: {self.mean_validation_f1()} "
                    f"--- Mean Running Time: {self.mean_running_time()}")
