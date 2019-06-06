import collections
import logging
from pathlib import Path
from typing import List, Dict

from comet_ml import Experiment

from datafc.transform.evaluation import Evaluator
from datafc.utils.logging import setup_logging

setup_logging("conf/logging.yaml")
logger = logging.getLogger(__name__)

ex = Experiment(api_key="vdIucsoZDhIfkUnOV5DBwjhe7", project_name="general", workspace="minhptx")

name_to_original_values: Dict[str, List[str]] = collections.defaultdict(list)
name_to_target_values: Dict[str, List[str]] = collections.defaultdict(list)
name_to_groundtruth_values: Dict[str, List[str]] = collections.defaultdict(list)

name_to_result: Dict[str, float] = {}
name_to_time: Dict[str, float] = {}

dataset = "sygus"
mapping_method = "sim"
string_similarity = "cosine"

ex.log_parameters({"dataset": dataset, "mapping_method": mapping_method, "string_similarity": string_similarity})

evaluator = Evaluator(mapping_method, string_similarity)
for sub_folder in list((Path("data") / f"{dataset}").iterdir()):
    # if sub_folder.name not in ["bd4"]:
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

    evaluator.run_top_k_experiment(
        sub_folder.name,
        name_to_original_values[sub_folder.name][:1000],
        name_to_target_values[sub_folder.name][:1000],
        name_to_groundtruth_values[sub_folder.name][:1000],
        3,
    )

    evaluator.report_file(ex, sub_folder.name)

    evaluator.report_dataset(ex)
