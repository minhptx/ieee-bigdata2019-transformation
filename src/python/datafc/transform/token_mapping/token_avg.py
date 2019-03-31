import logging
from typing import List, Tuple

from sklearn.model_selection import KFold

from datafc.ml.classification.multi_binary import MultiBinary
from datafc.repr.column import Column
from datafc.sim.column_sim import values_jaccard, ngram_jaccard, syntactic_sim
from datafc.syntactic.token import TokenData
from datafc.transform.operators import Operation, Constant
from datafc.transform.token_mapping.base import TokenMappingBaseModel

logger = logging.getLogger(__name__)


class TokenAvgMappingModel(TokenMappingBaseModel):
    def __init__(self):
        super().__init__()

    def score_operation(self, operation) -> float:
        if isinstance(operation, Constant):
            return 1
        transformed_column = Column(values=operation.transform())
        target_column = Column(values=operation.target_token.values)
        return (values_jaccard(transformed_column, target_column) + syntactic_sim(transformed_column,
                                                                                  target_column)) / 2.0

    def generate_candidate_functions(self, source_token: TokenData, target_token: TokenData):
        try:
            return max({op: self.score_operation(op) for op in
                        Operation.find_suitable_transformations(source_token, target_token)}.items(),
                       key=lambda x: x[1])
        except Exception as e:
            logger.error(e)
            return None, 0.0

    def train_scoring_model(self, example_patterns):
        pass
