import os
import streamlit as st

# LangChain Imports (Modern v0.3+)
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Page Configuration
st.set_page_config(
    page_title="Customer Support Knowledge Base AI",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Customer Support AI Assistant")
st.caption("Ask any question about our services. Answers are generated using our official knowledge base.")

# ------------------------------------------------------------------
# 1. API Key Handling
# ------------------------------------------------------------------
groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

if not groq_api_key:
    st.error("🔑 GROQ_API_KEY not found! Please configure it in `.streamlit/secrets.toml` or Streamlit Cloud Secrets.")
    st.stop()

# ------------------------------------------------------------------
# 2. Automated Knowledge Base & Vector Store Loading
# ------------------------------------------------------------------
DATA_FILE_PATH = "knowledge_base.txt"

@st.cache_resource
def initialize_vectorstore():
    """Loads the repo file, splits text, creates embeddings, and builds FAISS index once."""
    if not os.path.exists(DATA_FILE_PATH):
        st.error(f"Missing knowledge base file at `{DATA_FILE_PATH}`. Please push it to your GitHub repository.")
        return None

    # Load file from repository root
    loader = TextLoader(DATA_FILE_PATH, encoding="utf-8")
    documents = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)

    # Generate Embeddings & Index
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    return vectorstore

with st.spinner("Loading knowledge base..."):
    vectorstore = initialize_vectorstore()

if vectorstore is None:
    st.stop()

# ------------------------------------------------------------------
# 3. Chat Interface & Memory Setup
# ------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display past messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ------------------------------------------------------------------
# 4. RAG Execution via Modern LCEL
# ------------------------------------------------------------------
user_query = st.chat_input("Ask a question about tech support, refunds, or shipping...")

if user_query:
    # Render user prompt
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )

            # Initialize Groq LLM
            llm = ChatGroq(
                model="llama3-70b-8192",
                groq_api_key=groq_api_key,
                temperature=0.0
            )

            def format_docs(docs):
                return "\n\n".join([doc.page_content for doc in docs])

            # Anti-Hallucination Prompting
            prompt = ChatPromptTemplate.from_template(
                """You are an official Customer Support AI Assistant.
Answer the question based ONLY on the following retrieved context.
If the answer cannot be determined from the context, state:
"I am sorry, but I cannot find this information in our official support documentation."
Do NOT invent facts not present in the context.

Context:
{context}

Question: {question}

Answer:"""
            )

            # Modern LCEL Chain execution
            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            response = rag_chain.invoke(user_query)
            st.markdown(response)

            # Append response to history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
