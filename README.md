# aiasearch - AI Augmented Search

aiasearch returns data grounded in your documents.
It is an implementation of Retrieval Augmented Generation (RAG) and largely built on top of [Langchain](https://github.com/langchain-ai/langchain).
aiasearch is extensible allowing the use of different models for embedding and question answering.

With the initial upload Ollama is used for creating sentence embeddings for semantic search. 
Ollama and Anthropic are available for question answering.

## Getting Started
Set environment variables for connecting to providers [Ollama](#ollama), and if desired [Anthropic](#anthropic).

### Example: Building Permit Search
- Implemented in main.py
- Loads data from data.csv
- Creates a vector store for semantic search
- Creates a keyword store for keyword search
- Searches for documents using both stores
- Sends the question and documents to an LLM for question answering

## Ollama
Ollama must be installed in accessible location.
The default address is 127.0.0.1:11434.
This can be changed by setting the environment variable OLLAMA_HOST

## Anthropic
Anthropic requires setting the ANTHROPIC_API_KEY environment variable