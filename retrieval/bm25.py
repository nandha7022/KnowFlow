from rank_bm25 import BM25Okapi

# BM25Retriever class for keyword-based information retrieval using BM25 algorithm
class BM25Retriever:
    def __init__(self, docs):
        # Convert each document's content into a list of words (tokenized text)
        self.corpus = [doc.page_content.split() for doc in docs]

        # Initialize the BM25 model with the tokenized corpus
        self.bm25 = BM25Okapi(self.corpus)

        # Keep original document references for retrieval
        self.docs = docs

    def search(self, query, top_k=3):
        # Tokenize the query string into words
        query_tokens = query.split()

        # Get relevance scores between query and each document in the corpus
        scores = self.bm25.get_scores(query_tokens)

        # Pair each score with its corresponding document and sort by score (high to low)
        ranked = sorted(zip(scores, self.docs), key=lambda x: x[0], reverse=True)

        # Return the top_k most relevant documents
        return [doc for _, doc in ranked[:top_k]]
