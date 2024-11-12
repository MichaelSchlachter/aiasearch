import logging
import os
import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from aiasearch import stopwords
from aiasearch.data.vector_store import VectorStore
from aiasearch.log import PROJECT_NAME


class ChromaVectorStore(VectorStore):
    """
    Vector store for similarity searching a repository of text. The search result documents are intended to be fed
    to an LLM with the user's question and the answer to be found without in the documents.
    """

    def __init__(self, vectorstore_dir, overwrite=False):
        """
        Creates a vector store that wraps the Langchain Chroma implementation.
        Ollama and the nomic-embed-text model are used for generating word embeddings.
        The Ollama host is set using the "OLLAMA_HOST" environment variable.
        If the vectorstore_dir exists the existing vector store will be loaded.
        :param vectorstore_dir: A directory to save the vector store files.
        :param overwrite: Will first delete the vector store directory
        """
        self._logger = logging.getLogger(PROJECT_NAME)
        self._vectorstore = None
        self._vectorstore_dir = vectorstore_dir
        self._ollama_host = "127.0.0.1:11434"
        try:
            host: str = os.environ["OLLAMA_HOST"]
            if host is not None:
                self._ollama_host = host
        except KeyError:
            self._logger.info("key OLLAMA_HOST not found in environment variables")

        self._local_embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=self._ollama_host)
        if overwrite:
            self.delete_vectorstore()
        if os.path.exists(self._vectorstore_dir):
            self._vectorstore = Chroma(
                persist_directory=self._vectorstore_dir,
                embedding_function=self._local_embeddings
            )

        self.stop_words = stopwords.load_stop_words()

    def add_documents(self, documents: [Document]) -> None:
        """
        Adds a list of Documents to the vector store. If the store does not exist it will be created.
        :param documents: List of Langchain Document objects
        """
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        split_documents = text_splitter.split_documents(documents)
        if self._vectorstore is None:
            self._vectorstore = Chroma.from_documents(documents=split_documents, embedding=self._local_embeddings,
                                                  persist_directory=self._vectorstore_dir)
        else:
            self._vectorstore.add_documents(split_documents)

    def search_documents(self, text: str, k=10) -> list[Document]:
        """
        Performs a similarity search on the vector store using the provided text.
        :param text: Text to search for.
        :param k: Maximum number of results to return. default: 10
        :return: A list of matched documents.
        """

        # semantic search
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        filtered_question = ' '.join(filtered_words)
        search_result_documents: list[Document] = self._vectorstore.similarity_search(filtered_question, k=k)

        return search_result_documents

    def delete_vectorstore(self):
        """
        Deletes the vector store directory.
        """
        if os.path.exists(self._vectorstore_dir):
            shutil.rmtree(self._vectorstore_dir)

    def get_document_count(self) -> int:
        """
        :return: Number of documents
        """
        if self._vectorstore is None:
            return 0
        return len(self._vectorstore.get())
