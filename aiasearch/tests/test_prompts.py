from aiasearch.prompts.prompts import PromptManager

def test_list_available_tasks():
    llama_prompts = PromptManager("llama3.1")
    tasks = llama_prompts.list_available_tasks()
    assert "query_grounded" in tasks

def test_get_full_prompt():
    llama_prompts = PromptManager("llama3.1")
    prompts = llama_prompts.get_full_prompt(
        task="query_grounded"
    )
    assert "system" in prompts
    assert "user" in prompts

    assert True
