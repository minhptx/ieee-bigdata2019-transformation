import logging
from typing import List, Tuple, Dict

from datafc.ml.classification.multi_class import MultiClass
from datafc.syntactic.pattern import Pattern
from datafc.syntactic.token import REGEX_TYPES, RegexType, TokenData
from datafc.transform.operators import Operation
from datafc.transform.token_mapping.base import TokenMappingBaseModel

logger = logging.getLogger(__name__)


class TokenAttrMappingModel(TokenMappingBaseModel):
    def __init__(self):
        self.scoring_model: MultiClass = MultiClass[str](self.compute_feature)
        self.token_to_idx: Dict[TokenData, str] = {}

    @staticmethod
    def compute_feature(value):
        feature_vector = {}
        # print(value)
        for idx, (_, _, token) in enumerate(RegexType.find_all_tokens_of_types(REGEX_TYPES, value)):
            feature_vector["POS " + token.values[0]] = True
            feature_vector["POS_TYPE " + token.token_type.name] = True
            feature_vector["POS_POSITION %d" % token.position] = True
            feature_vector["LENGTH %d" % len(token.values[0])] = True

        return feature_vector

    def score_operation(self, operation: Operation) -> float:
        scores = [self.scoring_model.predict_proba(x)[self.token_to_idx[operation.target_token]] for x in
                  operation.transform()]

        return sum(scores) * 1.0 / len(scores)

    def generate_candidates(self, source_token: TokenData, target_token: TokenData):
        try:
            operation = max({op: self.score_operation(op) for op in
                        Operation.find_suitable_transformations(source_token, target_token)}.items(),
                       key=lambda x: x[1])
            logger.error(str(operation))
            return operation
        except Exception as e:
            logger.error(e)
            return None, 0.0

    def train_scoring_model(self, example_patterns: List[Pattern]):
        idx = 0
        label_cols: List[Tuple[str, dict]] = []

        logger.debug("Number of patterns: ", len(example_patterns))

        for idx1, pattern in enumerate(example_patterns):
            for token in pattern.tokens:
                self.token_to_idx[token] = str(idx)
                for value in token.values:
                    label_cols.append((str(idx), value))
                idx += 1
        self.scoring_model.train(label_cols)
