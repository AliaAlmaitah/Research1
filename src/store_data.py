from chromadb import Client  # Ensure this import is correct
from generate_embeddings import generate_embeddings, split_documents, load_pdfs
import os

def store_embeddings(chunks, embeddings):
    client = Client()  # Initialize the Client
    collection = client.create_collection("my_collection")  # Create or get a collection

    # Generate unique IDs for each chunk
    ids = [f"doc_{i}" for i in range(len(chunks))]

    # Convert embeddings from tensor to list
    embeddings_list = embeddings.tolist()  # Ensure this is a list

    # Add documents, embeddings, and IDs to the collection
    collection.add(documents=chunks, embeddings=embeddings_list, ids=ids)
    return client

if __name__ == "__main__":
    print("Loading PDFs...")
    directory = os.path.join(os.path.dirname(__file__), '../data')  # Correct relative path to your data directory
    documents = load_pdfs(directory=directory)
    chunks = split_documents(documents)
    
    print("Generating embeddings...")
    embeddings = generate_embeddings(chunks)
    
    print("Storing embeddings in ChromaDB...")
    client = store_embeddings(chunks, embeddings)
    
    print("Stored embeddings in ChromaDB.")
