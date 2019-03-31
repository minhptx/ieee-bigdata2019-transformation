import collections
import logging
from pathlib import Path
from typing import List, Dict

from datafc.transform.evaluation import Evaluator
from datafc.utils.logging import setup_logging

setup_logging("conf/logging.yaml")

logger = logging.getLogger(__name__)

dataset = "sygus"
name_to_original_values: Dict[str, List[str]] = collections.defaultdict(list)
name_to_target_values: Dict[str, List[str]] = collections.defaultdict(list)
name_to_groundtruth_values: Dict[str, List[str]] = collections.defaultdict(list)

name_to_result: Dict[str, float] = {}
name_to_time: Dict[str, float] = {}

evaluator = Evaluator()
for sub_folder in list((Path("data") / f"{dataset}").iterdir()):
    # if sub_folder.name not in ["lastname-long"]:
    #     continue
    logger.info("Scenario: %s" % sub_folder.name)
    for file in sub_folder.iterdir():
        with file.open(encoding="utf-8") as reader:

            for row in reader.readlines():
                row = row.encode("utf-8").decode("ascii", "ignore")
                if "input" in file.name:
                    name_to_original_values[sub_folder.name].append(row.strip())
                if "transformed" in file.name:
                    name_to_target_values[sub_folder.name].append(row.strip())
                if "groundtruth" in file.name:
                    name_to_groundtruth_values[sub_folder.name].append(row.strip())

            length = len(name_to_original_values[file.name])

    evaluator.run_top_k_experiment(sub_folder.name, name_to_original_values[sub_folder.name][:1000],
                                   name_to_target_values[sub_folder.name][:1000],
                                   name_to_groundtruth_values[sub_folder.name][:1000], 5)

    evaluator.report_file(sub_folder.name)

evaluator.report_dataset()
