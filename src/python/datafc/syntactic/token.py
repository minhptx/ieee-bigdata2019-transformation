import string
from abc import ABCMeta
from typing import Tuple, List, Optional

import regex as re


class TokenData:
    def __init__(self, token_type: "TokenType", position: int, length: int, values=None):
        if values is None:
            values = []
        self.token_type: TokenType = token_type
        self.position: int = position
        self.length: int = length
        self.values = values

    def __hash__(self):
        return hash(self.token_type) + hash(self.position) + hash(self.length)

    def __str__(self):
        return f"TokenData({self.token_type.name}, {self.position}, {self.length})"

    def __eq__(self, token_data: "TokenData") -> bool:
        if self.token_type == token_data.token_type and self.length == token_data.length:
            return True
        return False

    def is_matched(self, token_data: "TokenData") -> bool:
        return self.token_type == token_data.token_type and self.length == token_data.length

    def combine(self, token_data: "TokenData") -> Optional["TokenData"]:
        if self.is_matched(token_data):
            return TokenData(self.token_type, self.position, self.length, self.values + token_data.values)
        return None

    @staticmethod
    def get_basic_pattern(str_value: str) -> List["TokenData"]:
        masked_data: List["TokenData"] = [TokenData(StartToken, -1, -1, [""])]

        while str_value:
            for regex_type in BASIC_TYPES:
                match_result = re.match("^" + regex_type.regex, str_value)
                if match_result:
                    masked_data.append(TokenData(regex_type, -1, len(match_result.group()), [match_result.group()]))
                    str_value = str_value[len(match_result.group()) :]
                    break
        return masked_data

    @staticmethod
    def get_basic_tokens(str_value: str) -> List["TokenType"]:
        str_value = str_value.strip()
        masked_data: List["TokenType"] = [StartToken]
        while str_value:
            for regex_type in BASIC_TYPES:
                match_result = re.match("^" + regex_type.regex, str_value)
                if match_result:
                    masked_data.append(regex_type)
                    str_value = str_value[len(match_result.group()) :]
                    break
        return masked_data

    @staticmethod
    def get_extended_pattern(str_value: str, output_types) -> List["TokenData"]:
        masked_data: List["TokenData"] = [TokenData(StartToken, -1, -1, [""])]
        while str_value:
            for regex_type in output_types + REGEX_TYPES:
                match_result = re.match("^" + regex_type.regex, str_value)
                if match_result:
                    masked_data.append(TokenData(regex_type, -1, len(match_result.group()), [match_result.group()]))
                    str_value = str_value[len(match_result.group()) :]
                    break
        return masked_data


class TokenType(metaclass=ABCMeta):
    def __init__(self, name: str, regex: str, super_types=None):
        super().__init__()
        if super_types is None:
            super_types = []
        self._name: str = name
        self._regex: str = regex
        self.super_types: List["TokenType"] = super_types

    def __eq__(self, other: "TokenType"):
        return self.regex == other.regex and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def match(self, str_value: str):
        match = re.match(self.regex, str_value)
        if match:
            return True
        return False

    @property
    def name(self) -> str:
        return self._name

    @property
    def regex(self) -> str:
        return self._regex

    @regex.setter
    def regex(self, value: str):
        self._regex = value

    @property
    def super_types(self) -> List["TokenType"]:
        return self._super_types

    @super_types.setter
    def super_types(self, value):
        self._super_types = value

    @staticmethod
    def create_union_type(atomic_types: List["TokenType"]):
        regex = "|".join([f"({atomic_type.regex})" for atomic_type in atomic_types])
        return TokenType("", regex)

    @staticmethod
    def find_atomic_type(str_value: str) -> "TokenType":
        for atomic_type in BASIC_TYPES:
            if atomic_type.match(str_value):
                return atomic_type
        raise ValueError(f"Unrecognized type of value {str_value}")

    @staticmethod
    def find_all_matches_of_type(token_type: "TokenType", str_value: str) -> List[str]:
        return [match.group(0) for match in re.findall(token_type.regex, str_value)]

    @staticmethod
    def find_all_indices_of_type(token_type: "TokenType", str_value: str) -> List[str]:
        return [match.span() for match in re.findall(token_type.regex, str_value)]

    @staticmethod
    def find_all_tokens_of_type(token_type: "TokenType", str_value: str) -> List[Tuple[int, int, TokenData]]:
        return [
            (match.start(), match.end(), TokenData(token_type, index, match.end() - match.start(), [match.group(0)]))
            for index, match in enumerate(re.finditer(token_type.regex, str_value))
        ]

    @staticmethod
    def find_all_tokens_of_types(token_types: List["TokenType"], str_value: str) -> List[Tuple[int, int, TokenData]]:
        result = []
        for type_name in token_types:
            result.extend(TokenType.find_all_tokens_of_type(type_name, str_value))
        return result

    @staticmethod
    def find_all_matches(str_value: str) -> List[Tuple[int, int, TokenData]]:
        return TokenType.find_all_tokens_of_types(REGEX_TYPES, str_value)


Alphanum = TokenType("Alphanum", r"[\p{L}\p{N}]+")
Alnumspace = TokenType("Alnumspace", r"[\p{L}\p{N}\p{Z}]+")
Alphabet = TokenType("Alphabet", r"\p{L}+", [Alphanum])
Uppercase = TokenType("Uppercase", r"\p{Lu}+", [Alphabet])
Lowercase = TokenType("Lowercase", r"\p{Ll}+", [Alphabet])
# Digit = TokenType("Digit", r"((?<![\.\p{N}+])\p{N}+\.\p{N}+(?![.\p{N}]))|(\p{N}+)", [Alphanum])
Digit = TokenType("Digit", r"[\p{N}]+", [Alnumspace, Alnumspace])
Whitespace = TokenType("Whitespace", r"\p{Z}+")
StartToken = TokenType("^", re.escape("^"))
EndToken = TokenType("$", re.escape("$"))

BASIC_TYPES = [Uppercase, Lowercase, Digit, Whitespace, StartToken, EndToken]
REGEX_TYPES = [Alphanum, Alphabet, Uppercase, Lowercase, Digit, Whitespace, StartToken, EndToken]

for mark in string.punctuation:
    BASIC_TYPES.append(TokenType(f"{mark}", re.escape(mark)))
    REGEX_TYPES.append(TokenType(f"{mark}", re.escape(mark)))
