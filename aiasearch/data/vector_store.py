from abc import ABC, abstractmethod

from langchain_core.documents import Document


class VectorStore(ABC):
    """Interface for working with a vector store"""

    @abstractmethod
    def add_documents(self, documents: [Document]):
        """Add documents to a vector store"""
        pass

    @abstractmethod
    def search_documents(self, text: str):
        """Search documents in a vector store"""
        pass

    @abstractmethod
    def get_document_count(self) -> int:
        """Return the number of documents in the vector store"""
        pass