from query_database import query_database
from chromadb import Chroma
from langchain.embeddings import create_embedding_function

if __name__ == "__main__":
    db = Chroma()
    embedding_function = create_embedding_function("aws-bedrock")
    question = "What are the requirements for a liquid rocket engine?"
    response = query_database(question, db, embedding_function)
    print(f"Response: {response}")
