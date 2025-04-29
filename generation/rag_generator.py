# generation/rag_generator.py

# Import necessary libraries for OpenAI API and environment variable management
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file (contains OpenAI API key)
load_dotenv()

# Initialize OpenAI client with API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function: rank_documents
# Purpose: Rank documents based on semantic similarity to user query using embeddings
def rank_documents(docs, query):
    # Import sentence-transformers library for semantic similarity calculation
    from sentence_transformers import SentenceTransformer, util
    
    # Load the pre-trained embedding model "all-MiniLM-L6-v2"
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Generate embedding vector for the query
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Generate embedding vectors for all document texts
    doc_embeddings = model.encode([doc.page_content for doc in docs], convert_to_tensor=True)

    # Calculate cosine similarity between query embedding and document embeddings
    scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]

    # Pair each document with its similarity score
    scored_docs = list(zip(docs, scores))

    # Sort documents by their similarity scores in descending order
    ranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)

    # Print top 5 ranked documents for debugging purposes
    print("\n Ranked Documents:")
    for i, (doc, score) in enumerate(ranked_docs[:5]):
        print(f"Rank {i+1} | Score: {score.item():.4f}")
        print(f"Content Preview: {doc.page_content[:200]}...\n")

    # Return top 5 ranked documents
    return [doc for doc, _ in ranked_docs[:5]]

# Function: generate_answer
# Purpose: Generate a GPT-4 powered answer using the relevant documents and user query
def generate_answer(docs, query, chat_history=None, level="Mid", role="Data Analyst", raw_context=""):
    # If no pre-ranked context provided, rank documents based on the query
    if not raw_context:
        docs = rank_documents(docs, query)

    # Combine document texts into a single context string
    context = raw_context or "\n".join([doc.page_content for doc in docs])

    # Format chat history to preserve conversation context (if provided)
    history = "\n\n".join(f"User: {str(h['user'])}\nBot: {str(h['bot'])}" for h in chat_history) if chat_history else ""

    # Determine the response tone based on user experience level
    if level == "Junior":
        tone = "Use simple language and provide a beginner-friendly explanation."
    elif level == "Mid":
        tone = "Give a clear and detailed explanation with moderate technical depth."
    else:
        tone = "Provide an advanced, in-depth explanation with professional terminology and insights."

    # Build the prompt for GPT-4 combining history, context, instructions, and user query
    prompt = f"""{history}

Context:
{context}

Instruction: {tone} This answer is for a {level} {role}.

User: {query}
Bot:"""

    # Make API call to OpenAI's GPT-4 model with the constructed prompt
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    # Return the generated answer from GPT-4
    return response.choices[0].message.content.strip()
