from pathlib import Path
from typing import List, Tuple, Dict, TypeVar, Callable, Union

from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer

from datafc.ml.classification.base import BaseClassifier

T = TypeVar('T')

DUMMY_STR = "|#$%^&"


class MultiClass(BaseClassifier[T]):
    def __init__(self, feature_func: Callable[[T], List[float]]):
        self.labeled_data: List[Tuple[str, T]] = []
        self.model = RandomForestClassifier(n_estimators=100)
        self.feature_func = feature_func
        self.vectorizer = DictVectorizer()

    def save(self, path: Union[str, Path]):
        dump(self.model, str(path))

    def load(self, path: Union[str, Path]):
        self.model = load(str(path))

    def clear(self):
        self.labeled_data.clear()

    def create_feature_vectors(self, labeled_cols1: List[Tuple[str, T]]) -> Tuple[List[List[float]], List[str]]:
        vectors = []
        labels = []
        for label1, col1 in labeled_cols1:
            vectors.append(self.feature_func(col1))
            labels.append(label1)
        return vectors, labels

    def train(self, labeled_cols: List[Tuple[str, T]]):
        train_vectors, train_classes = self.create_feature_vectors(labeled_cols)
        train_vectors = self.vectorizer.fit_transform(train_vectors)
        self.model.fit(train_vectors, train_classes)

    def predict(self, labeled_col: T) -> str:
        train_vectors, _ = self.create_feature_vectors([(DUMMY_STR, labeled_col)])
        train_vectors = self.vectorizer.transform(train_vectors)
        return max(self.model.predict_proba(train_vectors), key=lambda x: x[1])

    def predict_proba(self, labeled_col: T) -> Dict[str, float]:
        train_vectors, _ = self.create_feature_vectors([(DUMMY_STR, labeled_col)])
        train_vectors = self.vectorizer.transform(train_vectors)

        labels = list(self.model.classes_)
        scores = list(self.model.predict_proba(train_vectors)[0])

        return dict(zip(labels, scores))
