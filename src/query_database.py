# Importing necessary libraries
from chromadb import Client
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# Function to query the database and generate an answer
def query_database(question, collection, embedding_function):
    # Step 1: Generate an embedding for the question using the embedding function
    query_embedding = embedding_function(question)

    # Step 2: Query the document collection using the generated embedding
    # This retrieves the top 5 most relevant document chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    # Step 3: Extract the relevant document chunks from the query results
    relevant_chunks = results['documents'][0]
    
    # Step 4: Construct a prompt for the language model using the relevant document chunks
    # The prompt includes the extracted information and the original question
    prompt = (
        "Given the following technical information extracted from research papers, "
        "provide a clear and detailed answer to the question: \n\n"
        + "\n".join(relevant_chunks) + "\n\n"
        "Question: " + question
    )

    # Step 5: Generate an answer using a Hugging Face language model (GPT-2 in this case)
    generator = pipeline('text-generation', model='gpt2', truncation=True)
    response = generator(prompt, max_length=300, temperature=0.7, top_p=0.9, num_return_sequences=1, pad_token_id=50256)
    
    # Step 6: Format the response with the generated answer and references
    # The response includes both the generated text and the original document chunks as references
    answer_with_references = {
        "answer": response[0]['generated_text'],
        "references": relevant_chunks
    }

    return answer_with_references

# Main execution block
if __name__ == "__main__":
    # Step 7: Initialize the ChromaDB client to manage document collections
    client = Client()
    collection = client.get_or_create_collection("documents")  # Access or create the "documents" collection

    # Step 8: Initialize the embedding function with a pre-trained model from Hugging Face
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Step 9: Define the question to query the database
    question = "What is a liquid rocket engine?"

    # Step 10: Call the query_database function and store the result
    result = query_database(question, collection, embedding_function.embed_query)

    # Step 11: Display the generated answer and the references
    print(f"Answer:\n{result['answer']}\n")
    print("References:")
    for idx, ref in enumerate(result['references'], start=1):
        print(f"Reference {idx}:\n{ref}\n")
