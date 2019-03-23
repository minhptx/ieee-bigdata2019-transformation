import collections
from pathlib import Path
from typing import List, Dict
import time

from datafc.transform.evaluation import Evaluator

dataset = "ijcai"
name_to_original_values: Dict[str, List[str]] = collections.defaultdict(list)
name_to_target_values: Dict[str, List[str]] = collections.defaultdict(list)
name_to_groundtruth_values: Dict[str, List[str]] = collections.defaultdict(list)

name_to_result: Dict[str, float] = {}
name_to_time: Dict[str, float] = {}

evaluator = Evaluator()
for sub_folder in list((Path("data") / f"{dataset}").iterdir()):
    if sub_folder.name not in ["1st_dimension"]:
        continue
    print("File: ", sub_folder.name)
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

    save_time = time.time()
    result = evaluator.calculate_tree_result(sub_folder.name, name_to_original_values[sub_folder.name][:1000],
                                             name_to_target_values[sub_folder.name][:1000],
                                             name_to_groundtruth_values[sub_folder.name][:1000])

    name_to_result[sub_folder.name] = result[2]
    name_to_time[sub_folder.name] = time.time() - save_time

    print(name_to_result)
    print(name_to_time)
    print(sum(name_to_result.values()) / len(name_to_result))
    print(sum(name_to_time.values()) / len(name_to_result))

    print("Validation accuracy", evaluator.validation_accuracy())
