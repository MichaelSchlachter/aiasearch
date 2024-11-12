from typing import Literal

from langchain_core.documents import Document

from aiasearch.models.anthropic_provider import AnthropicProvider
from aiasearch.models.ollama_provider import OllamaProvider
from aiasearch.models.provider import Provider


class ChatProvider:
    def __init__(self, provider_name: str, model_name: str = "", provider = None):
        """
        Use directly when passing in your own object that implements Provider.
        Otherwise, use ChatProvider.create_instance_by_name()

        Args:
            provider_name (str): The model's provider name.
            model_name (str): Name of the model.
            provider (Provider): A class that implements Provider
        """

        self._provider_name = provider_name
        self._model_name = model_name
        self._provider = provider

    @classmethod
    def create_instance_by_name(cls, provider_name: Literal["ollama", "anthropic"], model_name: str = ""):
        """
                Initializes the Llm object using the provided model name.

                Args:
                    provider_name (str): The model's provider name. Ollama is expected to be run locally
                        Must be one of the following values:
                            - ollama
                            - anthropic
                    model_name (str): Name of the model.

                Raises:
                    ValueError: If the provided provider_name is not recognized.
                """
        return cls(provider_name, model_name, ChatProvider.get_provider(provider_name, model_name))

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @provider_name.setter
    def provider_name(self, value: str):
        if not isinstance(value, str):
            raise TypeError("Name must be a string")
        self._provider_name = value

    @property
    def model_name(self) -> str:
        return self._model_name

    @model_name.setter
    def model_name(self, value: str):
        if not isinstance(value, str):
            raise TypeError("Name must be a string")
        self._model_name = value

    @staticmethod
    def get_provider(provider_name, model_name) -> Provider:
        """
        Uses the provider name to return either an Ollama, or Anthropic provider.
        :return: Provider of subtype OllamaProvider, or AnthropicProvider
        """
        if provider_name == "anthropic":
            return AnthropicProvider(model_name=model_name)
        return OllamaProvider(model_name=model_name)

    def query_text(self, text: str) -> str:
        """
        Uses the provided text to query the AI provider and return its response.
        :param text: Text to use in the user message of the AI prompt.
        :return: Message from AI
        """
        return self._provider.query_text(text)

    def query_grounded(self, text: str, documents: [Document]) -> str:
        return self._provider.query_grounded(text, documents)