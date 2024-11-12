import pandas as pd
from langchain_core.documents import Document

from aiasearch.data.loader import Loader


# noinspection GrazieInspection
class CsvLoader(Loader):
    """
    Class used to load data from a csv file.
    """

    def __init__(self):
        pass

    def load(self, file_path="", label_columns=None, combine_columns=None, metadata_columns=None, max_rows=-1):
        """
        Loads data from a csv file. The file must contain a header row with column names. Each column's value will be
        appended to the document's content.
        :param file_path: full path to file including file name
        :param label_columns: Dictionary of columns to map to labels at the beginning of the text.
        {"permitnumber": "Permit Number} will be added as "Permit Number: 1234"
        :param combine_columns: List of columns to load. An empty list will return all columns.
        :param metadata_columns: List of columns to add to the document's metadata.
        :param max_rows: Limit the number of rows to load. -1 will return all rows.
        :return A document list
        """

        if max_rows == -1:
            df = pd.read_csv(file_path, on_bad_lines="skip")
        else:
            df = pd.read_csv(file_path, on_bad_lines="skip", nrows=max_rows)

        if label_columns is None:
            label_columns = []
        if combine_columns is None:
            combine_columns = list(df.columns)
        if metadata_columns is None:
            metadata_columns = []

        documents = []
        for index, row in df.iterrows():
            page_content = ""
            for column_name in label_columns:
                page_content += str(label_columns[column_name]) + ": " + str(row[column_name]) + "\n"
            for column_name in combine_columns:
                page_content += str(row[column_name]) + " "
            page_content.rstrip(' ')
            metadata = {}
            for column_name in metadata_columns:
                metadata[column_name] = str(row[column_name])
            documents.append(Document(page_content=page_content, metadata=metadata))

        return documents
