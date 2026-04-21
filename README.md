# RAG Playground

This repository is a collection of Retrieval-Augmented Generation experiments built with LangChain, Streamlit, FAISS, OpenAI, Groq, Google Generative AI, Hugging Face embeddings, Pinecone hybrid search, Astra DB, and ObjectBox. It is structured as a learning sandbox rather than a single packaged application.

The repo contains:

- beginner and intermediate notebooks for document loading, chunking, embeddings, vector stores, and retrieval chains
- advanced notebook work for tool-using RAG agents
- multiple Streamlit demos that answer questions over web or PDF content
- experiments with several vector backends: FAISS, Chroma, Pinecone, Astra DB, and ObjectBox

## Repository Layout

| Path | Purpose |
| --- | --- |
| `Beginner Level/notebook/document.ipynb` | Intro notebook covering document loading from text, web, and PDF sources, then splitting and vectorization. |
| `Intermediate Level/retriever.ipynb` | Retriever-focused notebook using Chroma and FAISS with OpenAI embeddings and retrieval chains. |
| `Advanced Level/L1_Langchain/rag_agent.ipynb` | Agent-oriented RAG notebook combining retriever tools with Wikipedia and Arxiv tools. |
| `AstraDB_serverless_demo.ipynb` | Astra DB serverless RAG walkthrough using OpenAI embeddings and LangChain orchestration. |
| `RAG_w_HybridSearch/experiment.ipynb` | Pinecone hybrid search experiment combining dense embeddings with BM25 sparse retrieval. |
| `RAG_Groq/app.py` | Streamlit app that loads a Wikipedia page, embeds it with OpenAI, stores vectors in FAISS, and answers questions with Groq. |
| `RAG_Q&AChatbot/qa_app.py` | Streamlit PDF Q&A app using OpenAI embeddings, FAISS, and Groq over local Harry Potter PDFs. |
| `RAG_HuggingFace/app.py` | Streamlit PDF Q&A app using Hugging Face BGE embeddings, FAISS, and Groq. |
| `GoogleAI+Groq/app.py` | Streamlit PDF Q&A app using Google Generative AI embeddings with Groq generation and FAISS retrieval. |
| `ObjectBox/app.py` | Streamlit PDF Q&A app that swaps FAISS for ObjectBox as the vector store backend. |
| `main.py` | Placeholder root script; not part of the RAG demos. |

## What Each Demo Does

### Beginner notebook

The beginner notebook introduces core ingestion patterns:

- `TextLoader` for local text documents
- `WebBaseLoader` for web content
- `PyPDFLoader` for PDF content
- recursive chunking for long documents
- OpenAI embeddings and vector store concepts

The accompanying data folder includes sample local content used to demonstrate loaders.

### Intermediate notebook

The intermediate notebook moves from loading to retrieval. It builds vector stores with Chroma and FAISS, converts them into retrievers, and then uses LangChain retrieval chains to answer questions about a source document.

### Advanced notebook

The advanced notebook explores agentic RAG. It builds a retriever tool over LangChain documentation and combines it with Wikipedia and Arxiv tools so an agent can choose between retrieval and external knowledge sources.

### Streamlit apps

The Streamlit projects follow a shared pattern:

1. Load environment variables with `python-dotenv`
2. Ingest web or PDF data
3. Split documents with `RecursiveCharacterTextSplitter`
4. Generate embeddings with a provider-specific embedding model
5. Store chunks in a vector store
6. Retrieve top-k chunks and answer with Groq `llama-3.1-8b-instant`

The main differences are the embedding provider and vector backend:

- `RAG_Groq`: web content + OpenAI embeddings + FAISS
- `RAG_Q&AChatbot`: PDF content + OpenAI embeddings + FAISS
- `RAG_HuggingFace`: PDF content + Hugging Face BGE embeddings + FAISS
- `GoogleAI+Groq`: PDF content + Google Generative AI embeddings + FAISS
- `ObjectBox`: PDF content + OpenAI embeddings + ObjectBox

## Data Sources In The Repo

- Harry Potter PDF collections are used by the PDF-based Streamlit apps.
- `Beginner Level/data/` contains starter documents for document loader experiments.
- `Intermediate Level/data/` contains the source PDF used by the retriever notebook.
- `RAG_w_HybridSearch/bm25_values.json` stores fitted BM25 encoder values for the hybrid search experiment.

## Requirements

The repo currently mixes a `pyproject.toml`, `requirements.txt`, notebooks, and backend-specific integrations. It is best treated as an experimentation workspace, not a single lockstep production environment.

Base requirements visible in the repo include:

- Python 3.12+
- LangChain core packages
- `streamlit`
- `python-dotenv`
- `pypdf` and `pymupdf`
- `faiss-cpu`
- `chromadb`
- `groq`
- `langchain-groq`
- `langchain-openai`
- `sentence-transformers`
- `pinecone`, `pinecone-text`, `pinecone-notebooks`
- `langchain-objectbox`

Depending on which demo you want to run, you may also need integrations that are referenced in code or notebooks but not fully represented in the root dependency files, such as:

- `langchain-google-genai`
- `langchain-astradb`
- `cassio`
- `beautifulsoup4`
- `wikipedia`
- `arxiv`
- `langchainhub`

## Setup

Create and activate a virtual environment, then install dependencies.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you want to work from the project metadata instead:

```powershell
pip install -e .
```

For notebook-specific integrations, install missing extras as needed when imports fail.

## Environment Variables

Different demos expect different provider credentials.

```env
OPENAI_API_KEY=
GROQ_API_KEY=
GOOGLE_AI_API_KEY=
PINECONE_API_KEY=
HUGGINGFACE_API_KEY=
ASTRA_DB_API_ENDPOINT=
ASTRA_DB_APPLICATION_TOKEN=
USER_AGENT=
```

Notes:

- `OPENAI_API_KEY` is used by the OpenAI embedding demos and several notebooks.
- `GROQ_API_KEY` is required by all Streamlit apps that generate answers with Groq.
- `GOOGLE_AI_API_KEY` is required by `GoogleAI+Groq/app.py`.
- `PINECONE_API_KEY` and `HUGGINGFACE_API_KEY` are required by the hybrid search notebook.
- `ASTRA_DB_API_ENDPOINT` and `ASTRA_DB_APPLICATION_TOKEN` are required by the Astra DB notebook.
- `USER_AGENT` is useful for the advanced notebook because web-backed tools warn when it is missing.

## Running The Streamlit Apps

From the repository root, run one of the following:

```powershell
streamlit run .\RAG_Groq\app.py
streamlit run .\RAG_Q&AChatbot\qa_app.py
streamlit run .\RAG_HuggingFace\app.py
streamlit run .\GoogleAI+Groq\app.py
streamlit run .\ObjectBox\app.py
```

Expected app behavior:

- the Groq web demo builds vectors immediately from a Wikipedia article
- the PDF apps require pressing `Embed Documents` before asking questions
- retrieved chunks are shown in a Streamlit expander for debugging and transparency

## Notable Caveats

- The repo uses a mix of old and new LangChain APIs, so some notebooks may need dependency pinning or small import updates.
- `ObjectBox/app.py` includes a note and log file showing a compatibility issue between `langchain-objectbox` and the installed LangChain stack.
- The PDF-based apps now use direct `pypdf.PdfReader` loading and skip unreadable files or pages instead of failing the entire ingestion pass.
- At least one Harry Potter PDF in the dataset is unreadable, so skipped-file warnings are expected in PDF demos.
- The Astra DB notebook appears to contain inline environment assignments in notebook cells. Those values should be moved to environment variables before sharing or publishing the notebook.
- Root `README.md` and `ReadMe.txt` were empty when this documentation was generated.

## Suggested Learning Path

1. Start with `Beginner Level/notebook/document.ipynb` to understand loaders, chunking, and embeddings.
2. Move to `Intermediate Level/retriever.ipynb` to see retrievers and retrieval chains.
3. Try `RAG_Q&AChatbot/qa_app.py` or `RAG_HuggingFace/app.py` for an end-to-end interactive app.
4. Explore `RAG_w_HybridSearch/experiment.ipynb` for dense + sparse retrieval.
5. Finish with `Advanced Level/L1_Langchain/rag_agent.ipynb` for tool-augmented agents.

## Status

This repository is best described as a RAG experimentation workspace. It is useful for learning and comparison across providers and vector stores, but it is not yet standardized around one dependency set, one execution path, or one deployment target.
