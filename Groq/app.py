import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

from dotenv import load_dotenv
load_dotenv()

#Loading groq API key from .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if "vector" not in st.session_state:
    # st.session_state.embeddings = OllamaEmbeddings(model="langchain-embedding-v1")
    st.session_state.embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.getenv("OPENAI_API_KEY"))

    st.session_state.loader = WebBaseLoader("https://en.wikipedia.org/wiki/Artificial_intelligence")
    st.session_state.documents = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.texts = st.session_state.text_splitter.split_documents(st.session_state.documents[:50])
    st.session_state.vector = FAISS.from_documents(st.session_state.texts, st.session_state.embeddings)

st.title("Groq Retrieval Augmented Generation (RAG) Example")
llm=ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant")

prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant for answering questions about based only on the provided context. 
    Use the following retrieved documents to answer the question.
    <context>
    {context}
    </context>
    Question:{input}

    """
)

documents_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
retriever = st.session_state.vector.as_retriever(search_kwargs={"k": 4})
retrieval_chain = create_retrieval_chain(retriever, documents_chain)

prompt = st.text_input("Enter your question here:")
if prompt:
    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt})
    print("Time taken: ", time.process_time() - start)
    st.write(response['answer'])

    # With Streamlit Expander
    with st.expander("Retrieved Documents"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("---")