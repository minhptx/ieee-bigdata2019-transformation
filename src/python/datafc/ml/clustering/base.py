from abc import abstractmethod
from typing import Generic, TypeVar, List, Dict

from datafc.ml import DistanceMeasurable

T = TypeVar("T", bound=DistanceMeasurable)


class ClusteringModel(Generic[T]):
    @abstractmethod
    def fit(self, samples: List[T]):
        pass

    @abstractmethod
    def transform(self, samples: List[T]) -> Dict[T, List[T]]:
        pass

    @abstractmethod
    def fit_and_transform(self, samples: List[T]) -> Dict[T, List[T]]:
        pass
