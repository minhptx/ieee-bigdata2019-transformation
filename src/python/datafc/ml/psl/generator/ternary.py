from collections import defaultdict
from pathlib import Path
from typing import TypeVar, Generic, List, Callable, Dict, Tuple

T1 = TypeVar('T1')
T2 = TypeVar('T2')
T3 = TypeVar('T3')


class TernaryGenerator(Generic[T1, T2, T3]):
    def __init__(self, name_to_value1: Dict[str, T1], name_to_value2: Dict[str, T2], name_to_value3: Dict[str, T3],
                 sim_func: Callable[[T1, T2, T3], Dict[str, float]]):
        self.name_to_value1: Dict[str, T1] = name_to_value1
        self.name_to_value2: Dict[str, T2] = name_to_value2
        self.name_to_value3: Dict[str, T2] = name_to_value3
        self.sim_func: Callable[[T1, T2, T3], Dict[str, float]] = sim_func
        self.predicates: List[str] = []

    def generate_values(self) -> Dict[str, List[Tuple[str, str, str, float]]]:
        name_to_ids_sim_list: Dict[str, List[Tuple[str, str, str, float]]] = defaultdict(list)
        for name1, value1 in self.name_to_value1.items():
            for name2, value2 in self.name_to_value2.items():
                for name3, value3 in self.name_to_value3.items():
                    name_to_sim = self.sim_func(value1, value2, value3)
                    for name, sim_value in name_to_sim:
                        name_to_ids_sim_list[name].append(
                            (name1, name2, name3, sim_value))

        self.predicates = list(name_to_ids_sim_list.keys())
        return name_to_ids_sim_list

    def write_psl_files(self, output_path: Path, prefix: str):
        name_to_ids_sim_list: Dict[str, List[Tuple[str, str, str, float]]] = self.generate_values()

        for name in name_to_ids_sim_list.keys():
            with (output_path / f"{prefix}_{name}").open("w") as writer:
                for (id1, id2, id3, sim) in name_to_ids_sim_list[name]:
                    writer.write(f"{id1}\t{id2}\t{id3}\t{sim}\n")

    def write_psl_files_by_ids(self, ids: List[Tuple[str, str, str]], output_path: Path, prefix: str):
        name_to_ids_sim_list: Dict[str, List[Tuple[str, str, str, float]]] = self.generate_values()

        for name in name_to_ids_sim_list.keys():
            with (output_path / f"{prefix}_{name}").open("w") as writer:
                for (id1, id2, id3, sim) in name_to_ids_sim_list[name]:
                    if (id1, id2, id3) in ids:
                        writer.write(f"{id1}\t{id2}\t{id3}\t{sim}\n")
