import collections
from typing import Generic, TypeVar, List, Dict

from datafc.ml.clustering import ClusteringModel
from datafc.ml import DistanceMeasurable

import numpy as np

T = TypeVar('T', bound=DistanceMeasurable)


class Nonzero(ClusteringModel, Generic[T]):
    def __init__(self):
        self.seeds: List[T] = []

    def fit(self, samples: List[T]):
        self.seeds = []
        self.seeds.append(np.random.choice(samples))
        temp_samples = samples[:]
        while temp_samples:
            for sample in temp_samples:
                for seed in self.seeds:
                    if seed.similarity(sample) > 0:
                        break
                else:
                    self.seeds.append(sample)
                temp_samples.remove(sample)

        print("Seed: " + str([x.values for x in self.seeds]))

    def transform(self, samples: List[T]) -> Dict[T, List[T]]:
        clusters = collections.defaultdict(list)

        for sample in samples:
            for seed in self.seeds:
                if seed.similarity(sample) > 0:
                    clusters[seed].append(sample)
                    break

        return clusters

    def fit_and_transform(self, samples: List[T]) -> Dict[T, List[T]]:
        self.fit(samples)

        return self.transform(samples)
