from pathlib import Path
import os
from datafc.utils.logging import setup_logging
import logging

import numpy as np

setup_logging("conf/logging.yaml")

logger = logging.getLogger(__name__)
dataset = "sygus"

os.makedirs("flashfill")

for sub_folder in list((Path("data") / f"{dataset}").iterdir()):
    input_file = sub_folder / "input.csv"
    groundtruth_file = sub_folder / "groundtruth.csv"

    input_data_path = Path("flashfill") / "input.csv"
    output_data_path = Path("flashfill") / "output.csv"

    input_values = input_file.open("r").readlines()
    groundtruth_values = input_file.open("r").readlines()

    example_size = len(input_values) // 2

    sampled_indices = np.random.choice(range(len(input_values)), example_size)

    with input_data_path.open("w") as input_writer:
        with output_data_path.open("w") as output_writer:
            for index in range(len(input_values)):
                if index in sampled_indices:
                    input_writer.write(f'"{input_values[index]}","{groundtruth_values[index]}"\n')
                else:
                    input_writer.write(f'"{input_values[index]}",""\n')
                output_writer.write(f'"{groundtruth_values[index]}"')




