from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import List

from datafc.utils import CONFIG


class PSLModel(metaclass=ABCMeta):
    def __init__(self):
        self.rules: List[str] = []
        self.predicates: List[str] = []

    def add_rule(self, rule: str):
        self.rules.append(rule)

    def add_predicate(self, predicate: str):
        self.predicates.append(predicate)

    def write_rules(self):
        with (Path(CONFIG.get("psl.model_path")) / "rules.txt").open("w") as writer:
            writer.writelines(self.rules)

    def write_predicates(self):
        with (Path(CONFIG.get("psl.model_path")) / "predicates.txt").open("w") as writer:
            writer.writelines(self.predicates)

    @abstractmethod
    def generate_prob_files(self, ):
        pass

    def generate_model(self):
        self.generate_prob_files()
