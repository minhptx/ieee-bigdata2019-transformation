from collections import defaultdict
from pathlib import Path
from typing import TypeVar, Generic, List, Callable, Tuple, Dict

T = TypeVar('T')


class UnaryGenerator(Generic[T]):
    def __init__(self, name_to_value: Dict[str, T], property_func: Callable[[T], Dict[str, float]]):
        self.name_to_value = name_to_value
        self.property_func = property_func
        self.predicates: List[str] = []

    def generate_values(self):
        name_to_ids_sim_list: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for name, value in self.name_to_value:
            name_to_property = self.property_func(value)
            for feature_name, property_value in name_to_property.items():
                name_to_ids_sim_list[feature_name].append((name, property_value))
        self.predicates = list(name_to_ids_sim_list.keys())
        return name_to_ids_sim_list

    def write_psl_files(self, output_path: Path, prefix: str):
        name_to_id_property_list: Dict[str, List[Tuple[str, str, float]]] = self.generate_values()

        for name in name_to_id_property_list.keys():
            with (output_path / f"{prefix}_{name}.txt").open("w") as writer:
                for (id1, sim) in name_to_id_property_list[name]:
                    writer.write(f"{id1}\t{sim}\n")

    def write_psl_files_by_ids(self, ids: List[str], output_path: Path, prefix: str):
        name_to_id_property_list: Dict[str, List[Tuple[str, str, float]]] = self.generate_values()

        for name in name_to_id_property_list.keys():
            with (output_path / f"{prefix}_{name}.txt").open("w") as writer:
                for (id1, sim) in name_to_id_property_list[name]:
                    if id1 in ids:
                        writer.write(f"{id1}\t{sim}\n")
