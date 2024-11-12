from langchain_core.documents import Document

from aiasearch.models.chat_provider import ChatProvider
from aiasearch.models.ollama_provider import OllamaProvider


class TestChatProvider:
    def test_ollama(self):
        chat_provider = ChatProvider.create_instance_by_name("ollama", "llama3.1")
        assert chat_provider.provider_name == "ollama"
        assert chat_provider.model_name == "llama3.1"
        response_text = chat_provider.query_text("Tell me you love me")
        assert "love" in response_text

    def test_ollama_with_provider(self):
        ollama_provider = OllamaProvider("llama3.1")
        chat_provider = ChatProvider("ollama", "llama3.1", ollama_provider)
        assert chat_provider.provider_name == "ollama"
        assert chat_provider.model_name == "llama3.1"
        response_text = chat_provider.query_text("Tell me you love me")
        assert "love" in response_text

    def test_ollama_grounded(self):
        ollama_provider = OllamaProvider("llama3.1")
        chat_provider = ChatProvider("ollama", "llama3.1", ollama_provider)
        assert chat_provider.provider_name == "ollama"
        assert chat_provider.model_name == "llama3.1"
        doc1 = Document(page_content="A100071 Change connector link layout from attached enclosed walkway to partially enclose walkway with canopy. 175 W Boundary RD,West Roxbury,MA")
        doc2 = Document(page_content="ALT1243901 Renovate existing kitchen , Demo finishes, New Electrical, Plumbing HVAC steam piping, Duct work, Install new finishes, Floors ,Ceilings and Paint.Health Department Review-#72""Renovate existing kitchen , Demo finishes, New Electrical, Plumbing HVAC steam piping, Duct work, Install new finishes, Floors ,Ceilings and Paint.Health Department Review-#72 175 W Boundary RD,West Roxbury,MA")
        documents: [Document] = [doc1, doc2]
        response_text = chat_provider.query_grounded("What work was performed for permit A100071", documents=documents)
        assert "connector" in response_text

    def test_anthropic(self):
        chat_provider = ChatProvider.create_instance_by_name("anthropic", "claude-3-5-haiku-20241022")
        assert chat_provider.provider_name == "anthropic"
        perform_test = False
        if perform_test:
            # The environment variable ANTHROPIC_API_KEY must be set or an exception is raised
            response_text = chat_provider.query_text("What is the capitol of New York state?")
            assert "Albany" in response_text

    def test_anthropic_no_api_key(self):
        chat_provider = ChatProvider.create_instance_by_name("anthropic", "claude-3-5-haiku-20241022")
        # The environment variable ANTHROPIC_API_KEY must be set or an exception is raised
        try:
            chat_provider.query_text("What is the capitol of New York state?")
        except TypeError:
            # langchain returns a TypeError when the API key is not correct
            assert True
