import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
import os

# Function: load_documents
# Purpose: Load all PDF files from a given folder and extract their content using PyPDFLoader
def load_documents(folder_path):
    docs = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            # Initialize the loader for each PDF
            loader = PyPDFLoader(os.path.join(folder_path, file))
            # Load the content and append to the documents list
            docs.extend(loader.load())
    return docs

# Function: split_documents
# Purpose: Split long documents into smaller chunks using a recursive text splitter
def split_documents(docs):
    # Configure chunk size and overlap for better context retention
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)
