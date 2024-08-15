# Importing necessary libraries
from sentence_transformers import SentenceTransformer  # Importing the SentenceTransformer for generating embeddings
from preprocess_data import split_documents, load_pdfs  # Importing functions to preprocess documents

# Function to generate embeddings for a list of text chunks
def generate_embeddings(chunks):
    # Load the pre-trained model 'all-MiniLM-L6-v2' from Sentence Transformers
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate embeddings for the chunks and convert the output to tensors for efficient computation
    embeddings = model.encode(chunks, convert_to_tensor=True)
    
    # Return the generated embeddings
    return embeddings

# Main execution block
if __name__ == "__main__":
    # Load documents from PDF files
    documents = load_pdfs()
    
    # Split the loaded documents into smaller chunks
    chunks = split_documents(documents)
    
    # Generate embeddings for the document chunks
    embeddings = generate_embeddings(chunks)
    
    # Print the number of generated embeddings
    print(f"Generated {len(embeddings)} embeddings.")
