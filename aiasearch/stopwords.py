"""
A collection of functions to work with stop words.

Stop words are common words to remove from text for the purpose of reducing their emphasis when searching
or classifying.
"""
import pathlib

def load_stop_words():
    """
    Load stop words from stopwords.txt
    :return: list: A list of stopwords
    """
    filename = "stopwords.txt"
    try:
        current_dir = pathlib.Path(__file__).resolve().parent
        file_path = current_dir / filename
        with open(file_path, 'r') as file:
            lines = file.readlines()
            return [line.strip() for line in lines]
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return []
