import collections
from pathlib import Path
from typing import List, Dict

import defaultlist
import regex as re

from datafc.syntactic import RegexType


def add_start_and_end_tokens(string_values: List[str]):
    for value in string_values:
        yield '^' + value + '$'


class Pattern:
    def __init__(self, regex_types: List[RegexType]):
        self.regex_sequence = regex_types

    def __eq__(self, other: 'Pattern'):
        return self.to_regex() == other.to_regex()

    def to_regex(self) -> str:
        return "".join([regex.regex for regex in self.regex_sequence])

    def __hash__(self):
        return hash(self.to_regex())


class PatternUnion:
    def __init__(self, patterns: List[Pattern]):
        self.patterns = patterns

    def to_regex(self) -> str:
        return r"(?:%s)" % "|".join(["(%s)" % pattern.to_regex() for pattern in self.patterns])


class RegexModel:
    def __init__(self, original_values: List[str], target_values: List[str]):
        self.original_values: List[str] = list(add_start_and_end_tokens(original_values))
        self.target_values: List[str] = list(add_start_and_end_tokens(target_values))

    def find_constants_on_target(self) -> List[Pattern]:
        token_to_count: Dict[Pattern, int] = collections.defaultdict(lambda: 0)
        for value in self.target_values:
            tokens = RegexType.find_all_matches(value)

        return constants_with_context_types

    @staticmethod
    def split_by_constant_patterns(patterns: List[Pattern], values: List[str]) -> List[List[str]]:
        split_string_values = defaultlist.defaultlist(lambda: [])
        pattern_union = PatternUnion(patterns)

        for idx, value in enumerate(values):
            print(value, pattern_union.to_regex(), re.split(pattern_union.to_regex(), value))
            split_string_values[idx].extend(re.split(pattern_union.to_regex(), value))

        return split_string_values

    @staticmethod
    def infer_variable_types(split_string_values: List[List[str]]):
        variable_types = set()
        for value in split_string_values:
            for element in value:
                variable_types.add(RegexType.find_atomic_type(element))
        return list(variable_types)

    def learn_target_variable_types(self):
        constant_patterns = self.find_constants_on_target()
        split_string_values = RegexModel.split_by_constant_patterns(constant_patterns, self.target_values)
        return RegexModel.infer_variable_types(split_string_values)

    @staticmethod
    def mask_variable_strings(string_values: List[str]):
        pass


if __name__ == "__main__":
    dataset = "wu-ijcai"
    for sub_folder in (Path("data") / f"{dataset}").iterdir():
        print(sub_folder.name)
        for file in sub_folder.iterdir():
            if "transformed" in file.name:
                with file.open() as reader:
                    lines = list(reader.readlines())
                    lines = [x.strip() for x in lines]
                    model = RegexModel([], lines)
                    print(model.learn_target_variable_types())
