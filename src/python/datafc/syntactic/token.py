from abc import ABCMeta, abstractmethod
from typing import Tuple, List, Optional

import regex as re


class TokenType(metaclass=ABCMeta):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def regex(self) -> str:
        pass

    @property
    @abstractmethod
    def super_types(self) -> List['TokenType']:
        pass

    def is_sub_type(self, token_type: 'TokenType'):
        return token_type in self.super_types


class TokenData:
    def __init__(self, token_type: TokenType, position: int, length: int, values=None):
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

    def __eq__(self, token_data: 'TokenData') -> bool:
        if self.token_type == token_data.token_type and self.length == token_data.length:
            return True
        return False

    def is_matched(self, token_data: 'TokenData') -> bool:
        return self.token_type == token_data.token_type

    def combine(self, token_data: 'TokenData') -> Optional['TokenData']:
        if self.is_matched(token_data):
            new_length = self.length if self.length == token_data.length else -1
            return TokenData(self.token_type, self.position, new_length, self.values + token_data.values)
        return None

    @staticmethod
    def get_basic_pattern(string: str) -> List['TokenData']:
        string = string.strip()
        masked_data: List['TokenData'] = [TokenData(StartToken, -1, -1, [""])]
        while string:
            for regex_type in BASIC_TYPES:
                # print(regex_type.regex, str)
                match_result = re.match("^" + regex_type.regex, string)
                if match_result:
                    masked_data.append(TokenData(regex_type, -1, len(match_result.group()), [match_result.group()]))
                    string = string[len(match_result.group()):]
                    break
        return masked_data

    @staticmethod
    def get_basic_tokens(string: str) -> List['TokenType']:
        string = string.strip()
        masked_data: List['TokenType'] = [StartToken]
        while string:
            for regex_type in BASIC_TYPES:
                match_result = re.match("^" + regex_type.regex, string)
                if match_result:
                    masked_data.append(regex_type)
                    string = string[len(match_result.group()):]
                    break
        return masked_data

    @staticmethod
    def get_pattern(string: str) -> List['TokenData']:
        masked_data: List['TokenData'] = [TokenData(StartToken, -1, -1, [""])]
        while string:
            for regex_type in REGEX_TYPES:
                # print(regex_type.regex, str)
                match_result = re.match("^" + regex_type.regex, string)
                if match_result:
                    masked_data.append(TokenData(regex_type, -1, len(match_result.group()), [match_result.group()]))
                    string = string[len(match_result.group()):]
                    break
        return masked_data


class RegexType(TokenType, metaclass=ABCMeta):
    def __init__(self, name: str, regex: str, super_types=None):
        super().__init__()
        if super_types is None:
            super_types = []
        self._name: str = name
        self._regex: str = regex
        self.super_types: List[TokenType] = super_types

    def __eq__(self, other: 'RegexType'):
        return self.regex == other.regex

    def __hash__(self):
        return hash(self.name)

    def match(self, string: str):
        match = re.match(self.regex, string)
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
    def super_types(self) -> List[TokenType]:
        return self._super_types

    @super_types.setter
    def super_types(self, value):
        self._super_types = value

    @staticmethod
    def find_atomic_type(string: str) -> 'RegexType':
        for atomic_type in ATOMIC_TYPES:
            if atomic_type.match(string):
                return atomic_type
        else:
            return Special(string)

    @staticmethod
    def find_all_matches_of_type(token_type: 'RegexType', string: str) -> List[str]:
        return [match.group(0) for match in re.findall(token_type.regex, string)]

    @staticmethod
    def find_all_indices_of_type(token_type: 'RegexType', string: str) -> List[str]:
        return [match.span() for match in re.findall(token_type.regex, string)]

    @staticmethod
    def find_all_tokens_of_type(token_type: 'RegexType', string: str) -> List[Tuple[int, int, TokenData]]:
        return [(match.start(), match.end(),
                 TokenData(token_type, index, match.end() - match.start(), [match.group(0)]))
                for index, match in enumerate(re.finditer(token_type.regex, string))]

    @staticmethod
    def find_all_tokens_of_types(token_types: List['RegexType'], string: str) -> List[Tuple[int, int, TokenData]]:
        result = []
        for type_name in token_types:
            result.extend(RegexType.find_all_tokens_of_type(type_name, string))
        return result

    @staticmethod
    def find_all_matches(string: str) -> List[Tuple[int, int, TokenData]]:
        return RegexType.find_all_tokens_of_types(REGEX_TYPES, string)


class Special(RegexType):
    def __init__(self, string: str):
        super(Special, self).__init__(string, re.escape(string), REGEX_TYPES)

    def __hash__(self):
        return hash(self.regex)

    def __eq__(self, other: 'Special'):
        return self.name == other.name


Alphanum = RegexType("Alphanum", r"[\p{L}\p{N}]+")
Alnumspace = RegexType("Alnumspace", r"[\p{L}\p{N}\p{Z}]+")
Alphabet = RegexType("Alphabet", r"\p{L}+", [Alphanum])
Uppercase = RegexType("Uppercase", r"\p{Lu}+", [Alphabet])
Lowercase = RegexType("Lowercase", r"\p{Ll}+", [Alphabet])
# Digit = RegexType("Digit", r"((?<![\.\p{N}+])\p{N}+\.\p{N}+(?![.\p{N}]))|(\p{N}+)", [Alphanum])
Digit = RegexType("Digit", r"\p{N}+", [Alphanum])
Whitespace = RegexType("Whitespace", r"\p{Z}+")
Punctuation = RegexType("Punctuation", r"[\p{P}\p{S}]+")

BASIC_TYPES = [Uppercase, Lowercase, Digit, Whitespace, Punctuation]
REGEX_TYPES = [Alphanum, Alphabet, Uppercase, Lowercase, Digit, Whitespace, Punctuation]

StartToken = Special("^")
EndToken = Special("$")
ATOMIC_TYPES = [Uppercase, Lowercase, Digit, Whitespace]

# class Constant(TokenType):
#     def __init__(self, string: str):
#         self.value = string
#
#     @property
#     def name(self) -> str:
#         return f"Constant{self.value}"
#
#     @property
#     def regex(self) -> str:
#         return re.escape(self.value)
#
#     @property
#     def super_types(self) -> List['TokenType']:
#         return list()
#
#     @staticmethod
#     def find_all_matches(string: str, maximum_length: int) -> List[Tuple[int, int, TokenData]]:
#         matches = []
#         substr_to_pos = collections.defaultdict(lambda: 0)
#         for length in range(1, maximum_length + 1):
#             for i in range(len(string) - length):
#                 substring = string[i: i + length]
#                 substr_to_pos[substring] += 1
#                 matches.append((i, i + length, TokenData(Constant(substring), substr_to_pos[substring], length)))
#         return matches
