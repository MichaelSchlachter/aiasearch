import pathlib

from aiasearch.data.csv_loader import CsvLoader


def test_load_csv():
    current_dir = pathlib.Path(__file__).resolve().parent.parent
    file_path = current_dir / "data.csv"
    loader = CsvLoader()
    documents = loader.load(file_path=file_path,
                            label_columns=None,
                            combine_columns=["permitnumber", "worktype", "permittypedescr", "description", "comments",
                                             "address", "city", "state", "zip"],
                            metadata_columns=["permitnumber"],
                            max_rows=100)
    assert len(documents) > 0
    assert "A1000569" in documents[0].page_content
