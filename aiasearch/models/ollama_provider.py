import logging
import os

import langchain

from aiasearch.models.provider import Provider
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_ollama import ChatOllama

from ..log import PROJECT_NAME
from ..prompts.prompts import PromptManager


class OllamaProvider(Provider):
    """
    Connection provider to models hosted locally with Ollama.
    The Ollama host is set using the "OLLAMA_HOST" environment variable.
    """

    def __init__(self, model_name: str="llama3.1", prompt_path=None):
        self._logger = logging.getLogger(PROJECT_NAME)
        self._host = "127.0.0.1:11434"
        self._model_name = model_name
        self._prompt_path = prompt_path

        try:
            host: str = os.environ["OLLAMA_HOST"]
            if host is not None:
                self._host = host
        except KeyError:
            self._logger.info("key OLLAMA_HOST not found in environment variables")

        self._model = ChatOllama(
            model=model_name,
            temperature=0,
            base_url=self._host
        )

    def query_text(self, text) -> str:
        """
        Sends a prompt and returns the text result.
        :param text: Prompt message to send to Ollama.
        :return: Text response from the LLM.
        """
        summarize_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "",
                ),
                ("human", "{input}"),
            ]
        )
        summary_chain = summarize_prompt | self._model | StrOutputParser()

        response_text = summary_chain.invoke(
            {
                "input": text,
            }
        )

        self._logger.info("response text: " + response_text)
        return response_text

    def query_grounded(self, text: str, documents: [Document]) -> str:
        langchain.verbose = True
        langchain.debug = True
        def format_docs(docs):
            t = "\n\n".join(' '.join(document.page_content.split()) for document in docs)
            return t

        llama_prompts = PromptManager(model_name=self._model_name, prompt_path=self._prompt_path)
        prompts = llama_prompts.get_full_prompt(
            task="query_grounded"
        )

        prompt_final_document_summary = ChatPromptTemplate.from_template(
            prompts["system"] + " \n " + prompts["user"]
        )

        chain = prompt_final_document_summary | self._model | StrOutputParser()
        result_text = chain.invoke({"docs": format_docs(documents), "text": text} )
        return result_text

