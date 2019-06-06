from pathlib import Path
import os

from datafc.transform.evaluation import Evaluator
from datafc.utils.logging import setup_logging
import logging
import csv

import numpy as np

setup_logging("conf/logging.yaml")

logger = logging.getLogger(__name__)

evaluator = Evaluator()

dataset = "nyc"

input_data_path = Path("flashfill") / dataset / "input"
output_data_path = Path("flashfill") / dataset / "output"

for input_file in input_data_path.iterdir():
    output_file = output_data_path / input_file.name

    original_values = []
    transformed_values = []
    groundtruth_values = []

    csv_reader = csv.reader(input_file.open("r", encoding="utf-8"))
    for line in csv_reader:
        original_values.append(line[0])
        transformed_values.append(line[1])

    csv_reader = csv.reader(output_file.open("r", encoding="utf-8"))
    for line in csv_reader:
        groundtruth_values.append(line[0])

    evaluator.run_flashfill_experiment(input_file.name, original_values, transformed_values, groundtruth_values)
    evaluator.report_file(input_file.name)

evaluator.report_dataset()
