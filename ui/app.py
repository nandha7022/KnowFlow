
# System-level setup
import sys
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"  # Prevent Streamlit from reloading unnecessarily
from dotenv import load_dotenv
load_dotenv()  # Load .env file for secret keys

# Add the root directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import app modules
import streamlit as st
from utils.preprocess import load_documents, split_documents
from retrieval.bm25 import BM25Retriever
from retrieval.faiss_vector import FAISSRetriever
from retrieval.hybrid_search import HybridSearch
from generation.rag_generator import generate_answer
from agents.role_agent import filter_docs_by_role_and_level

# LangChain LLM and vector tools
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
import pandas as pd

# Streamlit page configuration
st.set_page_config(page_title="KnowFlow - Assistant", layout="wide")

# Custom CSS for UI style
st.markdown("""
    <style>
        html, body {
            background: linear-gradient(to top right, #f5f5fa, #e0e6f6);
            font-family: 'Segoe UI', sans-serif;
        }
        h1 {
            margin-top: -30px;
            text-align: center;
            color: #30344c;
            font-weight: 600;
            font-size: 48px;
        }
        .stChatMessage {
            background-color: rgba(255, 255, 255, 0.75);
            border-radius: 16px;
            padding: 1rem;
            margin-bottom: 10px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
        }
        .css-1cpxqw2, .stTextInput, .stSelectbox, .stRadio {
            background-color: rgba(255, 255, 255, 0.6) !important;
            border-radius: 12px;
            padding: 0.5rem;
            box-shadow: 0px 4px 8px rgba(0,0,0,0.05);
        }
        .stTextInput > div > div > input {
            padding: 10px;
            border: none;
            background-color: #ffffff;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# App Title
st.markdown("<h1>KnowFlow</h1>", unsafe_allow_html=True)

# Session initialization for chatbot memory and context
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.role = "Data Analyst"
    st.session_state.level = "Mid"
    st.session_state.feedback_log = []

# Sidebar role & level selection
with st.sidebar:
    st.session_state.role = st.selectbox("Select Role", ["Data Scientist", "Data Engineer", "Data Analyst"], index=2)
    st.session_state.level = st.selectbox("Experience Level", ["Junior", "Mid", "Senior"], index=1)
    st.markdown("---")
    # 📥 Export feedback data as Excel
    if st.button("📥 Export Feedback to Excel"):
        df = pd.DataFrame(st.session_state.feedback_log)
        df.to_excel("feedback_log.xlsx", index=False)
        st.success("Feedback exported as Excel!")

# User input (chat box)
query = st.chat_input("Ask something about your tools, data, or training...")

# FAISSRetriever defined inline for semantic search
class FAISSRetriever:
    def __init__(self, docs, index_path="vector_db_index"):
        self.embeddings = OpenAIEmbeddings()
        self.index_path = index_path

        # If FAISS index already saved, load it
        if os.path.exists(index_path):
            try:
                self.db = FAISS.load_local(index_path, self.embeddings)
                print(f"Loaded FAISS index from: {index_path}")
            except Exception as e:
                print(f"Failed to load FAISS index. Rebuilding. Reason: {e}")
                self.db = FAISS.from_documents(docs, self.embeddings)
                self.db.save_local(index_path)
        else:
            # Create and save FAISS vector database
            self.db = FAISS.from_documents(docs, self.embeddings)
            self.db.save_local(index_path)
            print(f"Saved new FAISS index to: {index_path}")

    def search(self, query, k=3):
        return self.db.similarity_search(query, k=k)

# On user query: start document retrieval & answer generation
if query:
    st.session_state.chat_history.append({"user": query})

    # Load and prepare the knowledge base
    raw_docs = load_documents("data/knowledge_base")
    docs = split_documents(raw_docs)

    # Filter docs based on user role & level
    filtered_docs = filter_docs_by_role_and_level(docs, st.session_state.role, st.session_state.level)

    # Initialize search engines
    bm25 = BM25Retriever(filtered_docs)
    faiss = FAISSRetriever(filtered_docs)
    hybrid = HybridSearch(bm25, faiss)

    # Define LangChain tool wrappers
    def tool_bm25_func(input):
        return "\n".join([doc.page_content for doc in bm25.search(input)])

    def tool_faiss_func(input):
        return "\n".join([doc.page_content for doc in faiss.search(input)])

    def tool_hybrid_func(input):
        return "\n".join([doc.page_content for doc in hybrid.search(input)])

    # Setup agent with tools
    tools = [
        Tool(name="BM25 Retriever", func=tool_bm25_func, description="Keyword-based search."),
        Tool(name="FAISS Retriever", func=tool_faiss_func, description="Semantic embedding search."),
        Tool(name="Hybrid Retriever", func=tool_hybrid_func, description="Combination of keyword and semantic.")
    ]

    # Use OpenAI's GPT-4 to choose the best retrieval strategy and generate answer
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)

    # Agent selects best retriever and returns relevant context
    tool_selected_context = agent.invoke(query)

    # Generate the final answer with chat history and role-based tone
    answer = generate_answer([], query, chat_history=st.session_state.chat_history[:-1],
                             level=st.session_state.level, role=st.session_state.role,
                             raw_context=tool_selected_context)

    # Save bot response in chat history
    st.session_state.chat_history[-1]["bot"] = answer

# Display chat bubbles and collect feedback
for i, entry in enumerate(st.session_state.chat_history):
    # User question
    st.markdown(f"<div class='stChatMessage'><strong>You:</strong><br>{entry['user']}</div>", unsafe_allow_html=True)

    # Bot answer
    if "bot" in entry:
        st.markdown(f"<div class='stChatMessage'><strong>KnowFlow:</strong><br>{entry['bot']}</div>", unsafe_allow_html=True)

        # Was it helpful?
        feedback = st.radio(
            f"Was this helpful? (Q{i+1})",
            ["Yes", "No"],
            index=0,
            key=f"feedback_{i}"
        )

        # Comment box
        comment = st.text_input(
            f"Additional comments for Q{i+1}:",
            placeholder="Type your suggestions or feedback here...",
            key=f"comment_{i}"
        )

        # Save full feedback (only if not already saved for this Q)
        if len(st.session_state.feedback_log) <= i:
            st.session_state.feedback_log.append({
                "question": entry["user"],
                "answer": entry["bot"],
                "feedback": feedback,
                "comment": comment,
                "role": st.session_state.role,
                "level": st.session_state.level
            })
