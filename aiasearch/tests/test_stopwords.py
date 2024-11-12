from aiasearch.stopwords import load_stop_words

def test_load_stop_words():
    assert "and" in load_stop_words()
    assert "alfresco" not in load_stop_words()
