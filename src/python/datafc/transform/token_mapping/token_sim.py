import logging
from dataclasses import dataclass
from typing import List, Tuple

from sklearn.model_selection import KFold

from datafc.ml.classification.multi_binary import MultiBinary
from datafc.repr.column import Column
from datafc.sim.column_sim import values_jaccard, syntactic_sim, ngram_jaccard, length_syntactic_sim
from datafc.syntactic.token import TokenData
from datafc.transform.operators import Operation, Constant
from datafc.transform.token_mapping.base import TokenMappingBaseModel

logger = logging.getLogger(__name__)


class TokenSimMappingModel(TokenMappingBaseModel):
    def __init__(self):
        self.scoring_model: MultiBinary = MultiBinary[Column](self.compute_sim)
        super().__init__()

    @staticmethod
    def compute_sim(col1: Column, col2: Column) -> List[float]:
        return [values_jaccard(col1, col2), ngram_jaccard(col1, col2, 1), ngram_jaccard(col1, col2, 2),
                length_syntactic_sim(col1, col2)]

    def score_operation(self, operation) -> float:
        if isinstance(operation, Constant):
            return 1
        transformed_column = Column(values=operation.transform())
        target_column = Column(values=operation.target_token.values)
        result = self.scoring_model.predict_similarity(transformed_column, target_column)
        return result

    def generate_candidate_functions(self, source_token: TokenData, target_token: TokenData):
        operation_candidates = Operation.find_suitable_transformations(source_token, target_token)
        if operation_candidates:
            return max({op: self.score_operation(op) for op in operation_candidates}.items(), key=lambda x: x[1])
        return None, 0.0

    def train_scoring_model(self, example_patterns_by_groups):
        labeled_pairs = []

        for example_patterns in example_patterns_by_groups:
            labeled_cols: List[Tuple[str, Column]] = []
            idx = 0
            for pattern in example_patterns:
                for token in pattern.tokens:
                    if len(token.values) <= 2:
                        continue
                    idx += 1

                    kf = KFold(n_splits=2)
                    for train_indices, test_indices in kf.split(token.values):
                        train_values = []
                        for index in test_indices:
                            train_values.append(token.values[index])
                        column = Column(str(idx), str(idx), train_values)
                        labeled_cols.append((str(idx), column))
            for label1, col1 in labeled_cols:
                for label2, col2 in labeled_cols:
                    labeled_pairs.append((col1, col2, label1 == label2))
            if len(labeled_pairs) > 100:
                break
        try:
            self.scoring_model.train_from_pairs(labeled_pairs)
            print(self.scoring_model.model.coef_)
        except Exception as e:
            logger.error(e)
