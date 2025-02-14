import chromadb
import os
from langchain_community.document_loaders import PyPDFLoader
import chromadb.utils.embedding_functions as embedding_functions
from typing import List, Tuple
from fastapi import UploadFile
from google.api_core.exceptions import InvalidArgument

# Define the persist directory and collection name.
PERSIST_DIRECTORY = ".chroma"
COLLECTION_NAME ="vectorDB"

def delete_vector_collection(chroma_client, collection_name: str) -> Tuple[bool, str]:
    """
    Deletes an existing collection using the native Chroma PersistentClient.
    Returns (True, message) if deletion was successful; otherwise, (False, error message).
    """
    try:
        chroma_client.delete_collection(name=collection_name)

        return True, f"Collection '{collection_name}' deleted successfully."
        
    except Exception as e:
        return False, f"Unable to delete collection: {e}"

async def create_vector_db(uploaded_files: List[UploadFile], gemini_api_key: str, thread_id: str) -> Tuple[bool, str]:
    """
    Creates a new vector database from uploaded PDF files.
    Deletes the old collection (if it exists) using PersistentClient,
    then loads and splits the PDFs, builds embeddings, and persists the DB.
    
    Parameters:
        uploaded_files: List of Streamlit uploaded file objects.
    
    Returns:
        (True, success_message) or (False, error_message)
    """
    if not uploaded_files:
        return False, "No documents uploaded! Please upload PDFs first."

    chroma_client = chromadb.PersistentClient(path=".chroma")
  

    # If the persist directory exists, delete the previous collection.
    if f"{thread_id}_{COLLECTION_NAME}" in chroma_client.list_collections():
        success, msg = delete_vector_collection(chroma_client,f"{thread_id}_{COLLECTION_NAME}")
        if not success:
            return False, msg

    try:  
        google_ef  = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=gemini_api_key)
    except InvalidArgument:
        return False, "Invalid Gemini API Key"
    collection = chroma_client.create_collection(name=f"{thread_id}_{COLLECTION_NAME}", embedding_function=google_ef)


    # Process the uploaded files.
    cumulative_pages = []
    for uploaded_file in uploaded_files:
        # Save the uploaded file temporarily.
        file_path =  f"temp_{uploaded_file.filename}" 
        with open(file_path, "wb") as f:
            f.write(await uploaded_file.read())
        # Load and split the PDF into pages.
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        cumulative_pages.extend(pages)
        # Clean up the temporary file.
        os.remove(file_path)

    documents=[]
    metadatas=[]
    ids=[]

    for i,doc in enumerate(cumulative_pages):
        documents.append(doc.page_content)
        metadatas.append({"source":doc.metadata["source"],"page":doc.metadata["page"]})
        ids.append(str(i))

    collection.add(documents=documents,ids=ids,metadatas=metadatas)

    return True, f"Vector DB created and persisted"

