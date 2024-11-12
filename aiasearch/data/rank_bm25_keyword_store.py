import logging

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from aiasearch.data.keyword_store import KeywordStore
from aiasearch.log import PROJECT_NAME


class RankBm25KeywordStore(KeywordStore):
    def __init__(self):
        self._retriever = None
        self._logger = logging.getLogger(PROJECT_NAME)

    def add_documents(self, documents: [Document]):
        self._retriever = BM25Retriever.from_documents(documents)

    def search_documents(self, text: str, k=10):
        self._retriever.k = k
        documents = self._retriever.invoke(text)
        return documents

