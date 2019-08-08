import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Union, Any

from datafc.trans.model import TransformationModel

logger = logging.getLogger("myapp")


@dataclass
class Example:
    original_value: str
    groundtruth_value: str
    transformed_values: List[str]
    validation_result: str
    validation_components: List[Any]


@dataclass
class TopKTransformationResult:
    total_count: int = 0

    correct_count: int = 0
    top_k_correct_count: int = 0

    mrr_count: float = 0
    running_time: float = 0

    top_k_failed_cases: List[Example] = field(default_factory=list)
    top_k_failed_validations: List[Example] = field(default_factory=list)

    validation_tp_count: int = 0
    validation_tn_count: int = 0
    validation_fp_count: int = 0
    validation_fn_count: int = 0

    is_correct_transformation: int = 1


class Evaluator:
    def __init__(self, mapping_method="sim", mapping_features=None, with_flashfill=False, k=10):
        if mapping_features is None:
            mapping_features = ["jaccard", "syn"]
        self.name_to_result: Dict[str, TopKTransformationResult] = {}
        self.name_to_active_result: Dict[str, List[TopKTransformationResult]] = defaultdict(
            lambda: [TopKTransformationResult() for _ in range(30)]
        )

        self.k = k

        self.mapping_method = mapping_method
        self.with_flashfill = with_flashfill
        self.mapping_features = mapping_features

    def micro_accuracy(self, name):
        return self.name_to_result[name].correct_count * 1.0 / self.name_to_result[name].total_count

    def micro_active_accuracy(self, name, step):
        return (
            self.name_to_active_result[name][step].correct_count
            * 1.0
            / self.name_to_active_result[name][step].total_count
        )

    def micro_top_k_accuracy(self, name):
        return self.name_to_result[name].top_k_correct_count * 1.0 / self.name_to_result[name].total_count

    def mrr_score(self, name):
        return self.name_to_result[name].mrr_count / self.name_to_result[name].total_count

    def exact_accuracy(self):
        return sum([x.is_correct_transformation for x in self.name_to_result.values()]) / len(self.name_to_result)

    def micro_mean_accuracy(self):
        return (
            sum([x.correct_count for x in self.name_to_result.values()])
            * 1.0
            / sum([x.total_count for x in self.name_to_result.values()])
        )

    def macro_mean_accuracy(self):
        return sum([self.micro_accuracy(name) for name in self.name_to_result]) / len(self.name_to_result)

    def macro_mean_active_accuracy(self, step):
        return sum([self.micro_active_accuracy(name, step) for name in self.name_to_result]) / len(self.name_to_result)

    def micro_mean_top_k_accuracy(self):
        return (
            sum([x.top_k_correct_count for x in self.name_to_result.values()])
            * 1.0
            / sum([x.total_count for x in self.name_to_result.values()])
        )

    def macro_mean_top_k_accuracy(self):
        return sum([self.micro_top_k_accuracy(name) for name in self.name_to_result]) / len(self.name_to_result)

    def micro_mrr_score(self):
        return (
            sum([x.mrr_count for x in self.name_to_result.values()])
            * 1.0
            / sum([x.total_count for x in self.name_to_result.values()])
        )

    def macro_mrr_score(self):
        return sum([self.mrr_score(name) for name in self.name_to_result]) / len(self.name_to_result)

    def correct_transformation_ratio(self):
        return sum([x.is_correct_transformation for x in self.name_to_result.values()]) * 1.0 / len(self.name_to_result)

    def validation_precision(self, name):
        result = self.name_to_result[name]
        if result.validation_tp_count == 0 and result.validation_fp_count == 0 and result.validation_fn_count == 0:
            return 1.0
        try:
            return result.validation_tp_count * 1.0 / (result.validation_tp_count + result.validation_fp_count)
        except ZeroDivisionError:
            return 0.0

    def validation_recall(self, name):
        result = self.name_to_result[name]
        if result.validation_tp_count == 0 and result.validation_fn_count == 0:
            return 1.0
        try:
            return result.validation_tp_count * 1.0 / (result.validation_tp_count + result.validation_fn_count)
        except ZeroDivisionError:
            return 0.0

    def validation_f1(self, name):
        precision = self.validation_precision(name)
        recall = self.validation_recall(name)
        try:
            return (2 * precision * recall) / (precision + recall)
        except ZeroDivisionError:
            return 0

    def mean_validation_precision(self):
        try:
            return (
                sum([x.validation_tp_count for x in self.name_to_result.values()])
                * 1.0
                / sum([x.validation_tp_count + x.validation_fp_count for x in self.name_to_result.values()])
            )
        except ZeroDivisionError:
            return 0

    def mean_validation_recall(self):
        try:
            return (
                sum([x.validation_tp_count for x in self.name_to_result.values()])
                * 1.0
                / sum([x.validation_tp_count + x.validation_fn_count for x in self.name_to_result.values()])
            )
        except ZeroDivisionError:
            return 0

    def mean_validation_f1(self):
        precision = self.mean_validation_precision()
        recall = self.mean_validation_recall()

        try:
            return (2 * precision * recall) / (precision + recall)
        except ZeroDivisionError:
            return 0

    def mean_running_time(self):
        return sum([x.running_time for x in self.name_to_result.values()]) / len(self.name_to_result)

    def check_validation_result(self, name, original_value, groundtruth_value, transformed_values, validation_result):
        if transformed_values[0] == groundtruth_value:
            if validation_result[0]:
                self.name_to_result[name].validation_fp_count += 1
                result = "FP"
                example = Example(original_value, groundtruth_value, transformed_values, result, validation_result[1:])
                self.name_to_result[name].top_k_failed_validations.append(example)
            else:
                self.name_to_result[name].validation_tn_count += 1
                result = "TN"
        else:
            if validation_result[0]:
                self.name_to_result[name].validation_tp_count += 1
                result = "TP"
            else:
                self.name_to_result[name].validation_fn_count += 1
                result = "FN"
                example = Example(original_value, groundtruth_value, transformed_values, result, validation_result[1:])
                self.name_to_result[name].top_k_failed_validations.append(example)

        return result

    def check_transformation_result(
        self,
        name: str,
        original_to_groundtruth_values: Dict[str, str],
        original_to_transformed_values: List[Tuple[str, List[str]]],
    ):
        logger.debug("Original to groundtruth %s" % original_to_groundtruth_values)
        logger.debug("Original to transformed %s" % original_to_transformed_values)

        for (original_value, transformed_value, validated_result) in original_to_transformed_values:
            self.name_to_result[name].total_count += 1

            validation_result = self.check_validation_result(
                name,
                original_value,
                original_to_groundtruth_values[original_value],
                transformed_value,
                validated_result,
            )

            if transformed_value == original_to_groundtruth_values[original_value]:
                self.name_to_result[name].correct_count += 1
            else:
                self.name_to_result[name].is_correct_transformation = 0

    def check_top_k_transformation_result(
        self,
        name: Union[str, Tuple[str, int]],
        original_to_groundtruth_values: Dict[str, str],
        original_to_k_transformed_values: List[Tuple[str, List[str], List[bool]]],
    ):
        assert isinstance(name, str), "Nonactive learning result requires name as string"
        tran_result = self.name_to_result[name]

        for (original_value, transformed_values, validated_results) in original_to_k_transformed_values:
            tran_result.total_count += 1
            transformed_values = transformed_values[: self.k]

            validation_result = self.check_validation_result(
                name,
                original_value,
                original_to_groundtruth_values[original_value],
                transformed_values,
                validated_results[0],
            )

            if transformed_values[0] == original_to_groundtruth_values[original_value]:
                tran_result.correct_count += 1
                tran_result.top_k_correct_count += 1
                tran_result.mrr_count += 1
            else:
                example = Example(
                    original_value,
                    original_to_groundtruth_values[original_value],
                    transformed_values,
                    validation_result,
                    [True, True]
                )
                tran_result.top_k_failed_cases.append(example)
                logger.debug(
                    "Failed case: %s vs '%s' (source: '%s')"
                    % (transformed_values[:3], original_to_groundtruth_values[original_value], original_value)
                )
                try:
                    correct_rank = transformed_values.index(original_to_groundtruth_values[original_value])
                except ValueError:
                    correct_rank = -1
                if correct_rank != -1:
                    tran_result.mrr_count += 1.0 / (correct_rank + 1)
                    tran_result.top_k_correct_count += 1
                else:
                    tran_result.is_correct_transformation = 0

    def run_flashfill_experiment(
        self, name, original_values: List[str], transformed_values: List[str], groundtruth_values: List[str]
    ):
        self.name_to_result[name] = TopKTransformationResult()

        original_to_groundtruth_values: Dict[str, str] = {}
        original_to_transformed_values: List[Tuple[str, List[str], List[bool]]] = []

        for original_value, transformed_value, groundtruth_value in zip(
            original_values, transformed_values, groundtruth_values
        ):
            original_to_groundtruth_values[original_value] = groundtruth_value
            original_to_transformed_values.append((original_value, [transformed_value], [True]))

        self.check_top_k_transformation_result(name, original_to_groundtruth_values, original_to_transformed_values)

    def run_normal_experiment(
        self, name: str, original_values: List[str], target_values: List[str], groundtruth_values: List[str]
    ):

        self.name_to_result[name] = TopKTransformationResult()

        original_to_groundtruth_values: Dict[str, str] = {}

        for index, value in enumerate(original_values):
            original_to_groundtruth_values[value] = groundtruth_values[index]

        starting_time = time.time()

        transformation_model = TransformationModel(self.mapping_method, self.mapping_features)
        validated_original_to_transformed_values, scores = transformation_model.learn(original_values, target_values)

        self.name_to_result[name].running_time = time.time() - starting_time

        assert len(original_values) == len(validated_original_to_transformed_values), (
            f"Dataset sizes before and after transformation "
            f"should be the same ({len(original_values)} vs {len(validated_original_to_transformed_values)}) "
        )

        self.check_transformation_result(name, original_to_groundtruth_values, validated_original_to_transformed_values)

    def run_top_k_experiment(
        self, name: str, original_values: List[str], target_values: List[str], groundtruth_values: List[str], k: int
    ):

        self.name_to_result[name] = TopKTransformationResult()

        original_to_groundtruth_values: Dict[str, str] = {}

        for index, value in enumerate(original_values):
            original_to_groundtruth_values[value] = groundtruth_values[index]

        starting_time = time.time()

        transformation_model = TransformationModel(self.mapping_method, self.mapping_features)

        validated_original_to_k_transformed_values, scores = transformation_model.learn_top_k(
            original_values, target_values, k
        )

        self.name_to_result[name].running_time = time.time() - starting_time
        assert len(original_values) == len(validated_original_to_k_transformed_values), (
            f"Dataset sizes before and after transformation "
            f"should be the same ({len(original_values)} vs {len(validated_original_to_k_transformed_values)}) "
        )

        self.check_top_k_transformation_result(
            name, original_to_groundtruth_values, validated_original_to_k_transformed_values
        )

    def generate_scenario_report(self, name):
        report = {
            "name": name,
            "micro_acc": self.micro_accuracy(name),
            "micro_top_k_acc": self.micro_top_k_accuracy(name),
            "mrr_score": self.mrr_score(name),
            "validation_p": self.validation_precision(name),
            "validation_r": self.validation_recall(name),
            "validation_f1": self.validation_f1(name),
            "running_time": self.name_to_result[name].running_time,
            "failed_transformation": self.name_to_result[name].top_k_failed_cases,
            "failed_validation": self.name_to_result[name].top_k_failed_validations,
        }

        return report

    def generate_dataset_report(self, name):
        report = {
            "name": name,
            "macro_mean_acc": self.macro_mean_accuracy(),
            "micro_mean_acc": self.micro_mean_accuracy(),
            "macro_top_k_acc": self.macro_mean_top_k_accuracy(),
            "micro_top_k_acc": self.micro_mean_top_k_accuracy(),
            "validation_p": self.mean_validation_precision(),
            "validation_r": self.mean_validation_recall(),
            "validation_f1": self.mean_validation_f1(),
            "mean_running_time": self.mean_running_time(),
            "correct_transformations":  self.correct_transformation_ratio()
        }

        return report

    def run_scenario(self, scenario_folder):
        original_values = []
        target_values = []
        groundtruth_values = []

        for file in scenario_folder.iterdir():
            with file.open(encoding="utf-8") as reader:

                for row in reader.readlines():
                    row = row.encode("utf-8").decode("ascii", "ignore")
                    if "input" in file.name:
                        original_values.append(row.strip())
                    if "transformed" in file.name:
                        target_values.append(row.strip())
                    if "groundtruth" in file.name:
                        groundtruth_values.append(row.strip())

        self.run_top_k_experiment(
            scenario_folder.name, original_values[:1000], target_values[:1000], groundtruth_values[:1000], 10
        )

        scenario_report = self.generate_scenario_report(scenario_folder.name)
        return scenario_report

    def run_dataset(self, dataset_folder):
        scenario_to_report = {}

        for scenario_folder in dataset_folder.iterdir():
            scenario_report = self.run_scenario(scenario_folder)
            scenario_to_report[scenario_folder.name] = scenario_report

        dataset_report = self.generate_dataset_report(dataset_folder.name)
        dataset_report["scenarios"] = scenario_to_report
        return dataset_report
