from pathlib import Path

import streamlit as st
import os

from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings

from pypdf import PdfReader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from langchain_classic.chains import create_retrieval_chain
# The problem is that langchain-objectbox 0.1.0 is 
# being used with an incompatible LangChain stack in our environment, especially langchain-core 0.1.53.

from langchain_objectbox.vectorstores import ObjectBox
import time

#Load all the environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

#Loading groq API key from .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DATA_DIR = Path(__file__).resolve().parent / "data_harry_potter"


def load_pdf_documents(data_dir: Path) -> tuple[list[Document], list[str]]:
    documents: list[Document] = []
    skipped_files: list[str] = []

    for pdf_path in sorted(data_dir.glob("*.pdf")):
        try:
            reader = PdfReader(str(pdf_path))
        except Exception:
            skipped_files.append(pdf_path.name)
            continue

        for page_number, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
            except Exception:
                continue

            page_text = page_text.strip()
            if not page_text:
                continue

            documents.append(
                Document(
                    page_content=page_text,
                    metadata={"source": str(pdf_path), "page": page_number},
                )
            )

    return documents, skipped_files

st.title("ObjectBox VectorsoreDB with Llama 3.1-8B-instant Example")

llm=ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant")

prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant for answering questions about 
    based only on the provided context. 
    Use the following retrieved documents to answer the question.
    <context>{context}</context>
    Question:{input}

    """
)

def vector_embeddings():
    if "vector" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings() #embedding model
        # st.session_state.loader = PyPDFDirectoryLoader(str(DATA_DIR), glob="**/*.pdf") #data ingestion
        # st.session_state.documents = st.session_state.loader.load() #document loading
        st.session_state.documents, st.session_state.skipped_files = load_pdf_documents(DATA_DIR) #document loading alternate
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(    #text chunk creation
            chunk_size=1000,
            chunk_overlap=200,
        )
        st.session_state.texts = st.session_state.text_splitter.split_documents( #text splitting
            st.session_state.documents[:20]
        )
        if st.session_state.skipped_files:
            st.warning(
                "Skipped unreadable PDF files: "
                + ", ".join(st.session_state.skipped_files)
            )

        if not st.session_state.texts:
            st.error("No readable PDF content was found to build the vector store.")
            st.stop()

        st.session_state.vector = ObjectBox.from_documents(
            st.session_state.texts,
            st.session_state.embeddings,
        )

if st.button("Embed Documents"):
    vector_embeddings()
    st.write("Documents embedded successfully!")


documents_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
retriever = st.session_state.vector.as_retriever(search_kwargs={"k": 6})
retrieval_chain = create_retrieval_chain(retriever, documents_chain)

prompt = st.text_input("Enter your question here:")


if prompt:
    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt})
    print("Time taken: ", time.process_time() - start)
    st.write(response['answer'])

    # With Streamlit Expander for metadata and source details of retrieved documents
    with st.expander("Retrieved Documents with Metadata and Source Details"):
        for i, doc in enumerate(response["context"]):
            st.write(f"Document {i+1}:")
            st.write(doc.page_content)
            st.write(f"Source: {doc.metadata.get('source', 'Unknown')}, Page: {doc.metadata.get('page', 'Unknown')}")
            st.write("------------------------------------------")



# st.write(f"Loaded {len(st.session_state.documents)} PDF pages/documents.")
# st.write(f"Created {len(st.session_state.texts)} text chunks.")


# HuggingFace PDF Embedding Example
# Loaded 61 PDF pages/documents.

# Created 142 text chunks.


