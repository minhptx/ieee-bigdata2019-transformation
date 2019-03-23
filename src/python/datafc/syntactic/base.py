from abc import abstractmethod, ABCMeta

from typing import List

from datafc.syntactic.token import TokenData


class ContainStringTokens(metaclass=ABCMeta):

    @property
    @abstractmethod
    def tokens(self) -> List[TokenData]:
        pass

    @property
    @abstractmethod
    def values(self) -> List[str]:
        pass
