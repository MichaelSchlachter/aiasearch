from abc import ABC, abstractmethod

from langchain_core.documents import Document


class Provider(ABC):
    """Interface for model providers such as Ollama and Anthropic."""

    @abstractmethod
    def query_text(self, text: str) -> str:
        """Generic LLM prompt. Sends a prompt to the provider and returns the response text."""
        pass

    @abstractmethod
    def query_grounded(self, text: str, documents: [Document]) -> str:
        """Sends a prompt to the provider with grounding data and returns the response text."""
        pass