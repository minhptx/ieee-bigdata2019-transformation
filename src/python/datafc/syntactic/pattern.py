from functools import reduce
from typing import List, Dict, Optional

from datafc.repr.column import Column
from datafc.syntactic.token import TokenData, Uppercase, Lowercase, Alphabet, Digit, Alphanum, Whitespace, Alnumspace


class Pattern:
    def __init__(self, token_sequence: List[TokenData], level=0):
        self.tokens: List[TokenData] = token_sequence
        self.level = level
        # print(len(self.tokens[0].values))
        self.values = ["" for _ in range(len(self.tokens[0].values))]

        for token in self.tokens:
            for idx in range(len(token.values)):
                self.values[idx] += token.values[idx]

    def __str__(self):
        return " ".join([str(token) for token in self.tokens])

    def __eq__(self, pattern: 'Pattern') -> bool:
        if len(self.tokens) != len(pattern.tokens):
            return False
        for idx in range(len(self.tokens)):
            if self.tokens[idx] != pattern.tokens[idx]:
                return False
        return True

    def combine(self, pattern: 'Pattern') -> 'Pattern':
        assert self == pattern
        new_tokens = []
        for idx, token in enumerate(self.tokens):
            new_tokens.append(token.combine(pattern.tokens[idx]))

        return Pattern(new_tokens, self.level)

    def to_cols(self):
        columns = []

        for idx, token in enumerate(self.tokens):
            column = Column(str(idx), str(idx), token.values)
            columns.append(column)

        return columns

    def __hash__(self):
        return hash("".join([str(token) for token in self.tokens]))

    @staticmethod
    def build_from_string(string: str) -> 'Pattern':
        tokens = TokenData.get_basic_pattern(string)

        return Pattern(tokens)

    def level_up(self) -> Optional['Pattern']:
        tokens: List[Optional[TokenData]] = [None] * len(self.tokens)
        for idx, token in enumerate(self.tokens):
            if self.level == 0:
                tokens[idx] = TokenData(token.token_type, -1, -1, token.values)
            elif self.level == 1:
                if token.token_type == Uppercase or token.token_type == Lowercase:
                    tokens[idx] = TokenData(Alphabet, -1, -1, token.values)
                else:
                    tokens[idx] = self.tokens[idx]
            elif self.level == 2:
                if token.token_type == Alphabet or token.token_type == Digit:
                    tokens[idx] = TokenData(Alphanum, -1, -1, token.values)
                else:
                    tokens[idx] = self.tokens[idx]
            elif self.level == 3:
                if token.token_type == Alphanum or token.token_type == Whitespace:
                    tokens[idx] = TokenData(Alnumspace, -1, -1, token.values)
                else:
                    tokens[idx] = self.tokens[idx]

        idx = 0
        removed_indices = []
        while idx < len(tokens):
            # print("Idx ", idx)
            run_idx = idx + 1
            while run_idx < len(tokens):
                # print("Run idx", run_idx)
                if tokens[run_idx].token_type == tokens[idx].token_type:
                    tokens[idx].values = [value + tokens[run_idx].values[i]
                                          for i, value in enumerate(tokens[idx].values)]
                    removed_indices.append(run_idx)
                    run_idx += 1
                else:
                    break
            idx = run_idx

        for remove_idx in reversed(removed_indices):
            del tokens[remove_idx]

        if tokens:
            return Pattern(tokens, self.level + 1)
        return None


class PatternNode:
    def __init__(self, value: Pattern, parent: 'PatternNode' = None, children=None):
        if children is None:
            children = []
        self.value: Pattern = value
        self.children: List['PatternNode'] = children
        self.parent = parent


class PatternTree:
    def __init__(self):
        self.node_in_layers: List[List[PatternNode]] = [[] for _ in range(0, 5)]

    def get_all_patterns(self):
        return reduce(list.__add__, self.node_in_layers, [])

    def get_patterns_by_layers(self, layers):
        return [pattern_node.value for pattern_node in
                reduce(list.__add__, [self.node_in_layers[layer] for layer in layers], [])]

    @staticmethod
    def build_from_strings(str_values: List[str]) -> 'PatternTree':
        pattern_tree = PatternTree()
        pattern_to_node: Dict[str, PatternNode] = {}

        for str_value in str_values:
            pattern = Pattern.build_from_string(str_value.strip())
            if str(pattern) in pattern_to_node:
                pattern_to_node[str(pattern)].value = pattern.combine(pattern_to_node[str(pattern)].value)
            else:
                pattern_tree.node_in_layers[0].append(PatternNode(pattern))

        pattern_to_node.clear()
        for i in range(1, 5):
            for pattern_node in pattern_tree.node_in_layers[i - 1]:
                parent_pattern = pattern_node.value.level_up()

                # print("Level", i, pattern_node.value.level, parent_pattern.level)
                if parent_pattern is None:
                    continue
                if str(parent_pattern) in pattern_to_node:
                    pattern_to_node[str(parent_pattern)].value = \
                        pattern_to_node[str(parent_pattern)].value.combine(parent_pattern)
                    pattern_to_node[str(parent_pattern)].children.append(pattern_node)
                else:
                    parent_node = PatternNode(parent_pattern)
                    parent_node.children.append(pattern_node)
                    pattern_tree.node_in_layers[i].append(parent_node)
                    pattern_to_node[str(parent_pattern)] = parent_node
            pattern_to_node.clear()

        return pattern_tree
