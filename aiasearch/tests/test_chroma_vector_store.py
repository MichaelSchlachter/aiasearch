from aiasearch.data.chroma_vector_store import ChromaVectorStore
from aiasearch.data.csv_loader import CsvLoader


def test_add_documents():
    file_path = "../data.csv"
    loader = CsvLoader()
    documents = loader.load(file_path=file_path,
                            label_columns=None,
                            combine_columns=["permitnumber", "worktype", "permittypedescr", "description",
                                             "comments", "address", "city", "state", "zip"],
                            metadata_columns=["permitnumber"],
                            max_rows=100)
    vectorstore_dir = "./tmp"
    chroma = ChromaVectorStore(vectorstore_dir=str(vectorstore_dir))
    chroma.add_documents(documents)
    search_documents = chroma.search_documents("Find records for work on State ST, Boston", 10)
    assert len(search_documents) > 0
    chroma.delete_vectorstore()


def test_search_documents():
    assert True
