import pytest
from query_database import query_database
from chromadb import Client
from langchain_huggingface.embeddings import HuggingFaceEmbeddings  # Adjusting the import path

@pytest.fixture
def setup_db():
    client = Client()  # Initialize the client correctly
    # Assuming you have a collection named "documents"
    collection = client.get_or_create_collection("documents")
    return collection

def test_query_database(setup_db):
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Using HuggingFaceEmbeddings
    question = "What is a liquid rocket engine?"
    response = query_database(question, setup_db, embedding_function.embed_query)
    assert "rocket" in response['answer'].lower()  # Adjusted to check the answer in the response

if __name__ == "__main__":
    pytest.main()
