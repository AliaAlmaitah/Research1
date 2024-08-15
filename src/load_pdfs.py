import os  # Importing the os module to interact with the operating system
from PyPDF2 import PdfReader  # Importing PdfReader from the PyPDF2 library to read PDF files

# Function to load all PDF files from a specified directory
def load_pdfs(directory="data/"):
    documents = []  # Initialize an empty list to store the text extracted from PDFs
    
    # Loop through all the files in the specified directory
    for filename in os.listdir(directory):
        # Check if the file is a PDF
        if filename.endswith(".pdf"):
            # Construct the full path to the PDF file
            path = os.path.join(directory, filename)
            
            # Create a PdfReader object to read the PDF file
            reader = PdfReader(path)
            
            # Initialize an empty string to store the text extracted from the PDF
            text = ""
            
            # Loop through each page in the PDF file
            for page in reader.pages:
                # Extract text from the page and add it to the text string
                text += page.extract_text()
            
            # Add the extracted text to the documents list
            documents.append(text)
    
    # Return the list of documents containing the extracted text
    return documents

# Main execution block
if __name__ == "__main__":
    # Load PDF documents from the default directory ("data/")
    documents = load_pdfs()
    
    # Print the number of documents loaded
    print(f"Loaded {len(documents)} documents.")
