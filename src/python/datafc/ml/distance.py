from abc import abstractmethod


class DistanceMeasurable:
    @abstractmethod
    def distance(self, other: "DistanceMeasurable") -> float:
        pass

    def similarity(self, other: "DistanceMeasurable") -> float:
        return 1.0 - self.distance(other)
