import logging

from aiasearch.data.chroma_vector_store import ChromaVectorStore
from aiasearch.data.csv_loader import CsvLoader
from aiasearch.data.rank_bm25_keyword_store import RankBm25KeywordStore
from aiasearch.models.ollama_provider import OllamaProvider
from aiasearch.models.anthropic_provider import AnthropicProvider
from log import PROJECT_NAME, log_initialize

if __name__ == "__main__":
    """
    this is an example of calling aiasearch
    set the environment variable OLLAMA_HOST to os Ollama if not running on the same host. 
    Use the correct IP similar to 127.0.0.1:11434
    set the environment variable ANTHROPIC_API_KEY if using Anthropic which can be enabled below.
    """

    log_initialize()
    logger = logging.getLogger(PROJECT_NAME)

    logger.info("Starting " + PROJECT_NAME)

    # load documents from CSV file to search against
    csv_loader = CsvLoader()
    file_path = "./data.csv"
    label_columns = {"permitnumber": "Permit Number",
                     "declared_valuation": "Valuation",
                     "worktype": "Work Type",
                     "issued_date": "Issued Date"}
    documents = csv_loader.load(file_path=file_path,
                                label_columns=label_columns,
                                combine_columns=["permittypedescr",
                                                 "description",
                                                 "comments",
                                                 "address", "city", "state", "zip"],
                                metadata_columns=["permitnumber"],
                                max_rows=10000)

    # create a vector store interface. set overwrite=True to recreate the vectorstore
    overwrite_vectorstore = False
    vectorstore_dir = "./tmp"
    vector_store = ChromaVectorStore(vectorstore_dir=str(vectorstore_dir), overwrite=overwrite_vectorstore)
    if vector_store.get_document_count() == 0:
        vector_store.add_documents(documents)

    # create a keyword store
    rank_bm25 = RankBm25KeywordStore()
    rank_bm25.add_documents(documents)

    # search the vector store
    question = "List work performed on Newbury ST"

    search_documents = vector_store.search_documents(question, 10)
    logger.info("Vector Search Documents: " + str(len(search_documents)))
    logger.info("\n\n**************************************\n")
    if len(search_documents) > 0:
        logger.info("First Vector Document:\n" + search_documents[0].page_content)
    logger.info("\n\n**************************************\n")

    # keyword search
    filtered_question = [word for word in question.split() if word.lower() not in vector_store.stop_words]
    keyword_documents = rank_bm25.search_documents(question, k=10)
    search_documents.extend(keyword_documents)
    logger.info("Keyword Search Documents: " + str(len(keyword_documents)))
    logger.info("\n\n**************************************\n")
    if len(keyword_documents) > 0:
        logger.info("First Keyword Document:\n" + keyword_documents[0].page_content)

    logger.info("\n**************************************\n")

    # ask the llm to provide an answer based on the question
    llm = OllamaProvider(model_name="llama3.1")
    # llm = AnthropicProvider()
    result_text = llm.query_grounded(question, search_documents)
    logger.info("Question: " + question)
    logger.info("Answer: " + result_text)
