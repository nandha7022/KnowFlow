from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

# FAISSRetriever uses vector similarity search for semantic retrieval
class FAISSRetriever:
    def __init__(self, docs):
        # Load pre-trained HuggingFace embedding model to convert text into vectors
        self.embeddings = HuggingFaceEmbeddings()

        # Create a FAISS vector database from the embedded documents
        self.db = FAISS.from_documents(docs, self.embeddings)

    def search(self, query, top_k=3):
        # Perform similarity search with the embedded query and return top_k most similar docs
        return self.db.similarity_search(query, k=top_k)
