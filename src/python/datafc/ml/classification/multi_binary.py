from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple, Dict, TypeVar, Generic, Callable, Union
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from datafc.ml.classification.base import BaseClassifier

T = TypeVar('T')

DUMMY_STR = "|#$%^&"


class MultiBinary(BaseClassifier[T]):
    def __init__(self, sim_func: Callable[[T, T], List[float]], method: str = "lr"):
        self.labeled_data: List[Tuple[str, T]] = []
        if method == 'random_forest':
            self.model = RandomForestClassifier()
        else:
            self.model = LogisticRegression()
        self.sim_func = sim_func

    def save(self, path: Union[str, Path]):
        dump(self.model, str(path))

    def load(self, path: Union[str, Path]):
        self.model = load(str(path))

    def store(self, labeled_cols: List[Tuple[str, T]]):
        self.labeled_data.extend(labeled_cols)

    def clear(self):
        self.labeled_data.clear()

    def create_feature_vectors(self, labeled_cols1: List[Tuple[str, T]],
                               labeled_cols2: List[Tuple[str, T]]) -> Tuple[List[List[float]], List[bool]]:
        vectors = []
        labels = []
        for label1, col1 in labeled_cols1:
            for label2, col2 in labeled_cols2:
                if col1 == col2:
                    continue
                vectors.append(self.sim_func(col1, col2))
                labels.append(label1 == label2)
        return vectors, labels

    def train(self, labeled_cols: List[Tuple[str, T]]):
        train_vectors, train_classes = self.create_feature_vectors(labeled_cols, labeled_cols)
        self.model.fit(train_vectors, train_classes)

    def predict(self, labeled_col: T) -> str:
        train_vectors, _ = zip(*self.create_feature_vectors([(DUMMY_STR, labeled_col)], self.labeled_data))
        return max(self.model.predict_proba(train_vectors), key=lambda x: x[1])

    def predict_proba(self, labeled_col: T) -> Dict[str, float]:
        train_vectors, _ = zip(*self.create_feature_vectors([(DUMMY_STR, labeled_col)], self.labeled_data))

        labels = [label for label, _ in self.labeled_data]
        scores = self.model.predict_proba(train_vectors)

        return dict(zip(*labels, scores))

    def predict_similarity(self, original_col: T, target_col: T) -> float:
        train_vectors, _ = self.create_feature_vectors([(DUMMY_STR, original_col)], [(DUMMY_STR, target_col)])
        return self.model.predict_proba(train_vectors)[0][1]
