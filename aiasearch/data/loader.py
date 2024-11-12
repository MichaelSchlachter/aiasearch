from abc import ABC, abstractmethod

from langchain_core.documents import Document


class Loader(ABC):
    """Interface for loading data into a list of type Document"""

    @abstractmethod
    def load(self) -> [Document]:
        """Load data from files and databases"""
        pass