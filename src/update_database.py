# Importing necessary functions
from store_data import store_embeddings  # Function to store embeddings in the database
from generate_embeddings import generate_embeddings  # Function to generate embeddings for text data

# Function to update the database with new document chunks
def update_database(db, new_chunks):
    # Loop through each new document chunk
    for chunk in new_chunks:
        # Check if the chunk already exists in the database using its unique ID
        if not db.exists(chunk.id):
            # If the chunk doesn't exist, add it to the database
            # Generate an embedding for the chunk and store it
            db.add_document(chunk, generate_embeddings([chunk])[0])

# Main execution block
if __name__ == "__main__":
    # Example usage:
    # Initialize the database by storing initial embeddings (this is a placeholder, replace with actual initialization)
    db = store_embeddings([], [])

    # List of new document chunks to be added to the database (replace with actual chunks)
    new_documents = ["new_document_chunk"]

    # Call the update_database function to add new chunks to the database
    update_database(db, new_documents)

    # Print a message indicating that the database has been updated
    print("Database updated with new chunks.")
