import collections
import random
from typing import List, TypeVar, Generic, Dict, Tuple

from datafc.ml.clustering import ClusteringModel
from datafc.ml import DistanceMeasurable

T = TypeVar("T", bound=DistanceMeasurable)


class MinDistanceSeeder(Generic[T]):
    def __init__(self, samples: List[T]):
        self.samples: List[T] = samples
        self.seeds: List[T] = []
        self.sample_to_distance: Dict[T, float] = {}
        self.sample_to_seed: Dict[T, T] = {}

    def find_seeds(self):
        last_distance = float("+inf")
        current_distance = float("+inf")

        while not (last_distance - current_distance < 0.1):
            last_distance = current_distance
            seed, distance = self.next_seed()
            current_distance = distance
            self.seeds.append(seed)
            self.update_mappings(seed)

        return self.seeds

    def find_nearest_seed(self, sample: T) -> Tuple[T, float]:
        return min([(seed, sample.distance(seed)) for seed in self.seeds], key=lambda x: x[1])

    def next_seed(self) -> Tuple[T, float]:
        if not self.seeds:
            return random.choice(self.samples), float("+inf")

        sample, distance = max(self.sample_to_distance.items(), key=lambda x: x[1])
        return sample, distance

    def update_mappings(self, seed: T):
        self.remove_sample(seed)
        for sample in self.samples:
            if sample in self.sample_to_distance:
                distance = sample.distance(seed)
                if distance < self.sample_to_distance[sample]:
                    self.sample_to_distance[sample] = distance
                    self.sample_to_seed[sample] = seed
            else:
                self.sample_to_distance[sample] = sample.distance(seed)
                self.sample_to_seed[sample] = seed

    def remove_sample(self, sample: T):
        self.samples.remove(sample)
        if sample in self.sample_to_distance:
            del self.sample_to_distance[sample]
        if sample in self.sample_to_seed:
            del self.sample_to_seed[sample]


class KMeansPlus(ClusteringModel[T]):
    def __init__(self):
        self.seeder = None

    def fit(self, samples: List[T]):
        self.seeder = MinDistanceSeeder(samples)
        self.seeder.find_seeds()

    def transform(self, samples: List[T]) -> Dict[T, List[T]]:
        clusters = collections.defaultdict(list)
        for sample in samples:
            seed = self.seeder.find_nearest_seed(sample)
            clusters[seed].append(samples)
        return clusters

    def fit_and_transform(self, samples: List[T]) -> Dict[T, List[T]]:
        self.fit(samples)
        clusters = collections.defaultdict(list)

        for sample in self.seeder.samples:
            clusters[self.seeder.sample_to_seed[sample]].append(sample)

        return clusters
