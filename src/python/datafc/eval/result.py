from dataclasses import dataclass, field
from enum import Enum
from typing import List, Union


@dataclass
class Example:
    original_value: str
    groundtruth_value: str
    transformed_values: List[str]
    validation_result: str
    validation_components: List[Union[bool, float]]


class ValidationResult(Enum):
    TP = 1
    FP = 2
    TN = 3
    FN = 4


@dataclass
class TopKTransformationResult:
    name: str = ""
    total_count: int = 0

    correct_count: int = 0
    top_k_correct_count: int = 0

    mrr_count: float = 0
    running_time: float = 0

    top_k_failed_transformations: List[Example] = field(default_factory=list)
    top_k_failed_validations: List[Example] = field(default_factory=list)

    validation_tp_count: int = 0
    validation_tn_count: int = 0
    validation_fp_count: int = 0
    validation_fn_count: int = 0

    transformation_result: bool = True

    validation_result: ValidationResult = ValidationResult.TN
