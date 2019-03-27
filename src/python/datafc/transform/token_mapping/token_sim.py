import logging
from typing import List, Tuple

from sklearn.model_selection import KFold

from datafc.ml.classification.multi_binary import MultiBinary
from datafc.repr.column import Column
from datafc.sim.column_sim import values_jaccard, ngram_jaccard, syntactic_sim
from datafc.syntactic.token import TokenData
from datafc.transform.operators import Operation
from datafc.transform.token_mapping.base import TokenMappingBaseModel

logger = logging.getLogger(__name__)


class TokenSimMappingModel(TokenMappingBaseModel):
    def __init__(self):
        self.scoring_model: MultiBinary = MultiBinary[Column](self.compute_sim)
        super().__init__()

    @staticmethod
    def compute_sim(col1: Column, col2: Column) -> List[float]:
        return [values_jaccard(col1, col2), syntactic_sim(col1, col2)]

    def score_operation(self, operation) -> float:
        original_column = Column(values=operation.transform())
        target_column = Column(values=operation.target_token.values)
        result = self.scoring_model.predict_similarity(original_column, target_column)
        return result

    def generate_candidates(self, source_token: TokenData, target_token: TokenData):
        try:
            return max({op: self.score_operation(op) for op in
                        Operation.find_suitable_transformations(source_token, target_token)}.items(),
                       key=lambda x: x[1])
        except Exception as e:
            logger.debug(e)
            return None, 0.0

    def train_scoring_model(self, example_patterns):
        idx = 0
        label_cols: List[Tuple[str, Column]] = []

        logger.debug("Number of patterns: ", len(example_patterns))

        for idx1, pattern in enumerate(example_patterns):
            for token in pattern.tokens:
                if len(token.values) <= 4:
                    continue
                idx += 1

                kf = KFold(n_splits=2)
                for train_indices, test_indices in kf.split(token.values):
                    train_values = []
                    for index in train_indices:
                        train_values.append(token.values[index])
                    column = Column(str(idx), str(idx), train_values)
                    label_cols.append((str(idx), column))
        try:
            self.scoring_model.train(label_cols)
        except Exception as e:
            logger.debug(e)
            # print("No training data")
            # self.scoring_model.model.coef_ = [0.5, 0.5]
