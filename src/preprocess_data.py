from load_pdfs import load_pdfs

def split_documents(documents, chunk_size=1000):
    chunks = []
    for document in documents:
        for i in range(0, len(document), chunk_size):
            chunk = document[i:i + chunk_size]
            chunks.append(chunk)
    return chunks

if __name__ == "__main__":
    documents = load_pdfs()
    chunks = split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
