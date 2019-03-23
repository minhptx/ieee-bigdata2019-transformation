from abc import abstractmethod, ABCMeta
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

from datafc.repr.column import Column
from datafc.ml.classification.multi_binary import MultiBinary
from datafc.syntactic.token import TokenData, Uppercase, Lowercase


class Operation(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, original_token: TokenData, target_token: TokenData, **kwargs):
        self.original_token = original_token
        self.target_token = target_token

    @abstractmethod
    def transform(self) -> List[str]:
        pass

    def score(self, model: MultiBinary) -> float:
        original_column = Column("original", "original", self.original_token.values)
        target_column = Column("target", "target", self.target_token.values)
        result = model.predict_similarity(original_column, target_column)
        # print("Prediction", self, result, self.original_values[:3], self.target_values[:3])
        return result

    @staticmethod
    def find_suitable_transformations(original_token: TokenData, target_token: TokenData) -> List['Operation']:
        transform_ops: List['Operation'] = []
        for Op in [Upper, Lower, Replace, Constant, Substring]:
            transform_ops.extend(Op.generate(original_token, target_token))
        return transform_ops

    @staticmethod
    def find_top_transformations(original_token: TokenData, target_token: TokenData, model: MultiBinary) -> \
            Tuple[Optional['Operation'], float]:
        try:
            return max({op: op.score(model) for op in
                        Operation.find_suitable_transformations(original_token, target_token)}.items(),
                       key=lambda x: x[1])
        except Exception:
            return None, 0.0

    @staticmethod
    def find_top_k_transformations(original_token: TokenData, target_token: TokenData, model: MultiBinary) -> \
            Dict['Operation', float]:
        return dict(sorted({op: op.score(model) for op in
                            Operation.find_suitable_transformations(original_token, target_token)}.items(),
                           key=lambda x: x[1], reverse=True)[:1])


class Substring(Operation):
    def __init__(self, original_token: TokenData, target_token: TokenData):
        super(Substring, self).__init__(original_token, target_token)
        self.start_index: int = 0
        self.length: int = -1
        self.best_score: float = 0

    def __str__(self):
        return f"Substring({self.start_index},{self.length})"

    @staticmethod
    def score_substring(original_values: List[str], target_values: List[str], model: MultiBinary) -> float:
        original_column = Column("original", "original", original_values)
        target_column = Column("target", "target", target_values)

        return model.predict_similarity(original_column, target_column)

    def find_best_parameters(self, model: MultiBinary):
        score_dict: Dict[int, float] = defaultdict(float)

        original_min_length = min([len(x) for x in self.original_token.values])
        target_length = len(self.target_token.values[0])

        for i in range(original_min_length - target_length + 1):
            value_list: List[str] = []
            for value in self.original_token.values:
                value_list.append(value[i: i + target_length])
            # print("Index list", i, value_list)
            score_dict[i] = Substring.score_substring(value_list, self.target_token.values, model)

            # print("Score ", i, value_list[:3], self.target_values[:3], score_dict[i])

            value_list = []
            for value in self.original_token.values:
                value_list.append(value[-i - target_length: -i])
            score_dict[-i] = Substring.score_substring(value_list, self.target_token.values, model)

            # print("Score ", -i, value_list[:3], self.target_values[:3], score_dict[i])
        if score_dict:
            self.start_index, self.best_score = max(score_dict.items(), key=lambda x: x[1])
        else:
            self.best_score = 0.0

        self.length = target_length

    def transform(self) -> List[str]:
        if self.length == -1:
            return [x[self.start_index:] for x in self.original_token.values]
        return [x[self.start_index: self.start_index + self.length] for x in self.original_token.values]

    def score(self, model: MultiBinary) -> float:
        if self.best_score == 0:
            self.find_best_parameters(model)
        return self.best_score

    @staticmethod
    def generate(original_token: TokenData, target_token: TokenData) -> List['Operation']:
        if len(set([len(x) for x in target_token.values])) != 1:
            return []
        return [Substring(original_token, target_token)]


class Upper(Operation):
    def __init__(self, original_token: TokenData, target_token: TokenData):
        super(Upper, self).__init__(original_token, target_token)

    def __str__(self):
        return "Upper"

    def transform(self) -> List[str]:
        return [x.upper() for x in self.original_token.values]

    @staticmethod
    def generate(original_token: TokenData, target_token: TokenData) -> List['Operation']:
        if original_token.token_type != Uppercase and target_token.token_type == Uppercase:
            return [Upper(original_token, target_token)]
        return []


class Lower(Operation):
    def __init__(self, original_token: TokenData, target_token: TokenData):
        super(Lower, self).__init__(original_token, target_token)

    def __str__(self):
        return "Lower"

    def transform(self) -> List[str]:
        return [x.lower() for x in self.original_token.values]

    @staticmethod
    def generate(original_token: TokenData, target_token: TokenData) -> List['Operation']:
        if original_token.token_type != Lowercase and target_token.token_type == Lowercase:
            return [Lower(original_token, target_token)]
        return []


class Replace(Operation):
    def __init__(self, original_token: TokenData, target_token: TokenData):
        super(Replace, self).__init__(original_token, target_token)

    def __str__(self):
        return "Replace"

    def transform(self) -> List[str]:
        return self.original_token.values

    @staticmethod
    def generate(original_token: TokenData, target_token: TokenData) -> List['Operation']:
        if target_token.token_type == original_token.token_type:
            # print(set([len(x) for x in target_token.values]))
            # print(set([len(x) for x in original_token.values]))
            # print(set([len(x) for x in target_token.values]).intersection(set([len(x) for x in original_token.values])))
            if len(set([len(x) for x in target_token.values]).intersection(
                    set([len(x) for x in original_token.values]))) != 0:
                return [Replace(original_token, target_token)]
        return []


class Constant(Operation):
    def __init__(self, original_token: TokenData, target_token: TokenData):
        super(Constant, self).__init__(original_token, target_token)

    def __str__(self):
        return f"Constant({self.target_token.values[0]})"

    def transform(self) -> List[str]:
        return [self.target_token.values[0] for _ in range(len(self.original_token.values))]

    @staticmethod
    def generate(original_token: TokenData, target_token: TokenData) -> List['Operation']:
        # print("Constant Token ", original_values[:5], len(set(target_values)) == 1)
        if len(set(target_token.values)) == 1 and len(target_token.values) != 1:
            return [Constant(original_token, target_token)]
        return []

    def score(self, model: MultiBinary):
        return 1.0
