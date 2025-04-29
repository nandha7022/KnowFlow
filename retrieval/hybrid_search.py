# HybridSearch combines BM25 (keyword) and FAISS (semantic) retrieval methods
class HybridSearch:
    def __init__(self, bm25, faiss):
        # Store initialized retriever objects
        self.bm25 = bm25
        self.faiss = faiss

    def search(self, query):
        # Run BM25 keyword-based search
        bm_docs = self.bm25.search(query)

        # Run FAISS semantic similarity search
        faiss_docs = self.faiss.search(query)

        # Combine both results and eliminate duplicates based on page content
        combined = list({d.page_content: d for d in bm_docs + faiss_docs}.values())

        # Return the top 3 unique documents
        return combined[:3]
