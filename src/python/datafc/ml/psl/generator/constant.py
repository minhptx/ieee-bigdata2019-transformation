from pathlib import Path
from typing import List, Dict, Tuple


class ConstantGenerator:
    def __init__(self, name_to_ids_value: Dict[str, List[Tuple[List[str], float]]]):
        self.name_to_ids_value: Dict[str, List[Tuple[List[str], float]]] = name_to_ids_value
        self.predicates: List[str] = list(self.name_to_ids_value.keys())

    def generate_values(self):
        pass

    def write_psl_files(self, output_path: Path, prefix: str):
        for name, ids_value_list in self.name_to_ids_value.items():
            with (output_path / f"{prefix}_{name}").open("w") as writer:
                for names, value in ids_value_list:
                    names_str = '\t'.join(names)
                    writer.write(f"{names_str}\t{value}\n")

    def write_psl_files_by_ids(self, ids: List[str], output_path: Path, prefix: str):
        raise NotImplementedError
