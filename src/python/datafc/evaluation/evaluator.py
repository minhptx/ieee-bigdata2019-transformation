import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Union

from datafc.transformation.model import TransformationModel

logger = logging.getLogger("myapp")


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
    def __init__(self, mapping_method="sim", mapping_features=None):
        if mapping_features is None:
            mapping_features = ["jaccard", "syn"]
        self.name_to_result: Dict[str, TopKTransformationResult] = {}
        self.name_to_active_result: Dict[str, List[TopKTransformationResult]] = defaultdict(
            lambda: [TopKTransformationResult() for _ in range(30)]
        )
        self.mapping_method = mapping_method
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
        return (
            self.name_to_result[name].top_k_correct_count
            * 1.0
            / self.name_to_result[name].total_count
        )

    def mrr_score(self, name):
        return self.name_to_result[name].mrr_count / self.name_to_result[name].total_count

    def exact_accuracy(self):
        return sum([x.is_correct_transformation for x in self.name_to_result.values()]) / len(
            self.name_to_result
        )

    def micro_mean_accuracy(self):
        return (
            sum([x.correct_count for x in self.name_to_result.values()])
            * 1.0
            / sum([x.total_count for x in self.name_to_result.values()])
        )

    def macro_mean_accuracy(self):
        return sum([self.micro_accuracy(name) for name in self.name_to_result]) / len(
            self.name_to_result
        )

    def macro_mean_active_accuracy(self, step):
        return sum([self.micro_active_accuracy(name, step) for name in self.name_to_result]) / len(
            self.name_to_result
        )

    def micro_mean_top_k_accuracy(self):
        return (
            sum([x.top_k_correct_count for x in self.name_to_result.values()])
            * 1.0
            / sum([x.total_count for x in self.name_to_result.values()])
        )

    def macro_mean_top_k_accuracy(self):
        return sum([self.micro_top_k_accuracy(name) for name in self.name_to_result]) / len(
            self.name_to_result
        )

    def micro_mrr_score(self):
        return (
            sum([x.mrr_count for x in self.name_to_result.values()])
            * 1.0
            / sum([x.total_count for x in self.name_to_result.values()])
        )

    def macro_mrr_score(self):
        return sum([self.mrr_score(name) for name in self.name_to_result]) / len(
            self.name_to_result
        )

    def validation_precision(self, name):
        result = self.name_to_result[name]
        if (
            result.validation_tp_count == 0
            and result.validation_fp_count == 0
            and result.validation_fn_count == 0
        ):
            return 1.0
        try:
            return (
                result.validation_tp_count
                * 1.0
                / (result.validation_tp_count + result.validation_fp_count)
            )
        except ZeroDivisionError:
            return 0.0

    def validation_recall(self, name):
        result = self.name_to_result[name]
        if result.validation_tp_count == 0 and result.validation_fn_count == 0:
            return 1.0
        try:
            return (
                result.validation_tp_count
                * 1.0
                / (result.validation_tp_count + result.validation_fn_count)
            )
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
                / sum(
                    [
                        x.validation_tp_count + x.validation_fp_count
                        for x in self.name_to_result.values()
                    ]
                )
            )
        except ZeroDivisionError:
            return 0

    def mean_validation_recall(self):
        try:
            return (
                sum([x.validation_tp_count for x in self.name_to_result.values()])
                * 1.0
                / sum(
                    [
                        x.validation_tp_count + x.validation_fn_count
                        for x in self.name_to_result.values()
                    ]
                )
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
        return sum([x.running_time for x in self.name_to_result.values()]) / len(
            self.name_to_result
        )

    def check_validation_result(
        self, name, validation_result, transformed_value, groundtruth_value
    ):
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

            self.check_validation_result(
                name,
                validated_result,
                transformed_value,
                original_to_groundtruth_values[original_value],
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
        k: int,
        is_active: bool,
    ):
        if is_active:
            assert isinstance(
                name, tuple
            ), "Active learning result requires name as tuple of (name, step)."
            tran_result = self.name_to_active_result[name[0]][name[1]]
        else:
            assert isinstance(name, str), "Nonactive learning result requires name as string"
            tran_result = self.name_to_result[name]

        for (
            original_value,
            transformed_values,
            validated_results,
        ) in original_to_k_transformed_values:
            tran_result.total_count += 1

            if not is_active:
                self.check_validation_result(
                    name,
                    validated_results[0],
                    transformed_values[0],
                    original_to_groundtruth_values[original_value],
                )

            if transformed_values[0] == original_to_groundtruth_values[original_value]:
                tran_result.correct_count += 1
                tran_result.top_k_correct_count += 1
                tran_result.mrr_count += 1
            else:
                logger.debug(
                    "Failed case: %s vs '%s' (source: '%s')"
                    % (
                        transformed_values[:3],
                        original_to_groundtruth_values[original_value],
                        original_value,
                    )
                )
                try:
                    correct_rank = transformed_values.index(
                        original_to_groundtruth_values[original_value]
                    )
                except ValueError:
                    correct_rank = -1
                if correct_rank != -1:
                    tran_result.mrr_count += 1.0 / (correct_rank + 1)
                    tran_result.top_k_correct_count += 1
                else:
                    tran_result.is_correct_transformation = 0

    def check_active_transformation_result(
        self,
        name,
        original_to_groundtruth_values,
        validated_original_to_transformed_pairs_by_pattern,
        scores_by_patterns,
        k,
        with_flashfill,
    ):
        validated_original_to_transformed_pairs_by_pattern.append([])

        for original_to_transformed_pairs in validated_original_to_transformed_pairs_by_pattern:
            self.check_top_k_transformation_result(
                name,
                original_to_groundtruth_values,
                original_to_transformed_pairs,
                k,
                is_active=False,
            )
            self.check_top_k_transformation_result(
                (name, 0),
                original_to_groundtruth_values,
                original_to_transformed_pairs,
                k,
                is_active=True,
            )

        def find_most_ambiguous_pattern(exceptions):
            best_pattern = None
            best_distance = float("inf")

            for pattern, scores in enumerate(scores_by_patterns):
                if pattern in exceptions:
                    continue
                if scores[0] == 0:
                    return pattern, 0
                if len(scores) <= 1:
                    continue
                top_2_score = sorted(scores, reverse=True)[:2]
                distance = top_2_score[0] - top_2_score[1]
                if distance < best_distance:
                    best_pattern = pattern
                    best_distance = distance
            return best_pattern, best_distance

        clarified_patterns = []

        for i in range(1, k):
            best_pattern_index, _ = find_most_ambiguous_pattern(clarified_patterns)
            if best_pattern_index is not None:
                clarified_patterns.append(best_pattern_index)
                best_pattern_string_pairs = validated_original_to_transformed_pairs_by_pattern[
                    best_pattern_index
                ]

                if best_pattern_string_pairs:
                    example_groundtruth = original_to_groundtruth_values[
                        best_pattern_string_pairs[0][0]
                    ]
                    if with_flashfill:
                        for idx in range(len(best_pattern_string_pairs)):
                            best_pattern_string_pairs[idx] = (
                                best_pattern_string_pairs[idx][0],
                                [original_to_groundtruth_values[best_pattern_string_pairs[idx][0]]],
                                best_pattern_string_pairs[idx][2],
                            )
                    else:
                        try:
                            chosen_index = best_pattern_string_pairs[0][1].index(
                                example_groundtruth
                            )
                        except ValueError:
                            chosen_index = -1
                        if chosen_index != -1:
                            for idx in range(len(best_pattern_string_pairs)):
                                best_pattern_string_pairs[idx] = (
                                    best_pattern_string_pairs[idx][0],
                                    # [best_pattern_string_pairs[idx][1][chosen_index]],
                                    [
                                        original_to_groundtruth_values[
                                            best_pattern_string_pairs[idx][0]
                                        ]
                                    ],
                                    best_pattern_string_pairs[idx][2],
                                )
                        else:
                            best_pattern_string_pairs[0] = (
                                best_pattern_string_pairs[0][0],
                                [original_to_groundtruth_values[best_pattern_string_pairs[0][0]]],
                                best_pattern_string_pairs[0][2],
                            )
                            clarified_patterns.remove(best_pattern_index)
                            validated_original_to_transformed_pairs_by_pattern[-1].append(
                                best_pattern_string_pairs[0]
                            )
                            del best_pattern_string_pairs[0]

            for original_to_transformed_pairs in validated_original_to_transformed_pairs_by_pattern:
                self.check_top_k_transformation_result(
                    (name, i),
                    original_to_groundtruth_values,
                    original_to_transformed_pairs,
                    k,
                    is_active=True,
                )

    def run_flashfill_experiment(
        self,
        name,
        original_values: List[str],
        transformed_values: List[str],
        groundtruth_values: List[str],
    ):
        self.name_to_result[name] = TopKTransformationResult()

        original_to_groundtruth_values: Dict[str, str] = {}
        original_to_transformed_values: List[Tuple[str, List[str], List[bool]]] = []

        for original_value, transformed_value, groundtruth_value in zip(
            original_values, transformed_values, groundtruth_values
        ):
            original_to_groundtruth_values[original_value] = groundtruth_value
            original_to_transformed_values.append((original_value, [transformed_value], [True]))

        self.check_top_k_transformation_result(
            name, original_to_groundtruth_values, original_to_transformed_values, 1, is_active=False
        )

    def run_normal_experiment(
        self,
        name: str,
        original_values: List[str],
        target_values: List[str],
        groundtruth_values: List[str],
    ):

        self.name_to_result[name] = TopKTransformationResult()

        original_to_groundtruth_values: Dict[str, str] = {}

        for index, value in enumerate(original_values):
            original_to_groundtruth_values[value] = groundtruth_values[index]

        starting_time = time.time()

        transformation_model = TransformationModel(self.mapping_method, self.mapping_features)
        validated_original_to_transformed_values, scores = transformation_model.learn(
            original_values, target_values
        )

        self.name_to_result[name].running_time = time.time() - starting_time

        assert len(original_values) == len(validated_original_to_transformed_values), (
            f"Dataset sizes before and after transformation "
            f"should be the same ({len(original_values)} vs {len(validated_original_to_transformed_values)}) "
        )

        self.check_transformation_result(
            name, original_to_groundtruth_values, validated_original_to_transformed_values
        )

    def run_top_k_experiment(
        self,
        name: str,
        original_values: List[str],
        target_values: List[str],
        groundtruth_values: List[str],
        k: int,
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
            name,
            original_to_groundtruth_values,
            validated_original_to_k_transformed_values,
            k,
            is_active=False,
        )

    def run_active_top_k_experiment(
        self,
        name: str,
        original_values: List[str],
        target_values: List[str],
        groundtruth_values: List[str],
        k: int,
        with_flashfill: bool,
    ):

        self.name_to_result[name] = TopKTransformationResult()

        original_to_groundtruth_values: Dict[str, str] = {}

        for index, value in enumerate(original_values):
            original_to_groundtruth_values[value] = groundtruth_values[index]

        starting_time = time.time()

        transformation_model = TransformationModel(self.mapping_method, self.mapping_features)
        validated_original_to_transformed_pairs_by_pattern, scores_by_patterns = transformation_model.learn_top_k_active(
            original_values, target_values, k
        )

        self.check_active_transformation_result(
            name,
            original_to_groundtruth_values,
            validated_original_to_transformed_pairs_by_pattern,
            scores_by_patterns,
            k,
            with_flashfill,
        )

        self.name_to_result[name].running_time = time.time() - starting_time

    def generate_scenario_report(self, name, k):
        report = {
            "name": name,
            "micro_acc": self.micro_accuracy(name),
            "micro_top_k_acc": self.micro_top_k_accuracy(name),
            "mrr_score": self.mrr_score(name),
            "validation_p": self.validation_precision(name),
            "validation_r": self.validation_recall(name),
            "validation_f1": self.validation_f1(name),
            "running_time": self.name_to_result[name].running_time,
        }

        last_value = -1
        active_accuracies = []
        for i in range(k):
            if self.micro_active_accuracy(name, i) != -1:
                last_value = self.micro_active_accuracy(name, i)
            active_accuracies.append(round(last_value, 2))

        report.update({f"active_learning_curve": active_accuracies})

        return report

    def generate_dataset_report(self, name, k):
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
        }

        last_value = -1
        active_accuracies = []

        for i in range(k):
            if self.macro_mean_active_accuracy(i) != -1:
                last_value = self.macro_mean_active_accuracy(i)
            active_accuracies.append(round(last_value, 2))

        report.update({f"active_learning_curve": active_accuracies})

        return report

    def run_scenario(self, scenario_folder, mapping_method, string_similarity, with_flashfill):
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

        self.run_active_top_k_experiment(
            scenario_folder.name,
            original_values[:1000],
            target_values[:1000],
            groundtruth_values[:1000],
            10,
            with_flashfill=with_flashfill,
        )

        scenario_report = self.generate_scenario_report(scenario_folder.name, 10)
        return scenario_report

    def run_dataset(self, dataset_folder, mapping_method, mapping_features, with_flashfill):
        scenario_reports = []

        for scenario_folder in dataset_folder.iterdir():
            scenario_report = self.run_scenario(
                scenario_folder, mapping_method, mapping_features, with_flashfill
            )
            scenario_reports.append(scenario_report)

        dataset_report = self.generate_dataset_report(dataset_folder.name, 10)
        dataset_report["scenarios"] = scenario_reports
        return dataset_report
