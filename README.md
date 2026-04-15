# RAG Playground

A practical, notebook-driven exploration of Retrieval-Augmented Generation (RAG), covering the progression from foundational document loading to traditional RAG, enhanced RAG, agentic RAG, and Typesense-backed retrieval.

This repository is intended as both a learning project and a working reference. It demonstrates how documents are ingested, chunked, embedded, indexed, retrieved, and passed into language models for grounded response generation.

## Overview

The project is organized around four complementary workflows:

- `notebook/document.ipynb` introduces LangChain `Document` objects and basic text loading.
- `notebook/pdf_loader.ipynb` implements a traditional and enhanced RAG pipeline over PDFs using local embeddings, ChromaDB, and Groq.
- `typesense.ipynb` explores search and retrieval with Typesense, both directly and through LangChain.
- `agentic-rag/1-agenticrag.ipynb` demonstrates a simple agentic RAG workflow built with LangGraph and conditional retrieval.

Together, these notebooks show how the same core RAG idea can be implemented with different retrieval backends, prompting strategies, and orchestration patterns.

## Architecture

### End-to-End RAG Flow

![RAG Flow](./RAG%20FlowChart.png)

At a high level, the project follows this sequence:

- user query
- query embedding
- retrieval from a vector or search backend
- prompt construction with retrieved context
- response generation by an LLM

### Traditional RAG

![Traditional RAG](./Traditional%20RAG.png)

The traditional pipeline used here follows the standard retrieve-then-generate pattern:

- a user submits a query
- relevant context is retrieved from a vector store
- the retrieved context is combined with a prompt
- an LLM generates the final response

### Agentic RAG

![Agentic RAG](./Agentic%20RAG.png)

The agentic version adds control flow and decision-making. Instead of always retrieving first, the system decides whether retrieval is needed and then routes execution accordingly. In practice, this introduces the kinds of choices agentic systems need to make:

- whether retrieval is needed
- what information should be retrieved
- where retrieval should happen
- how many retrieval steps are worth taking

## What the Project Covers

This repository demonstrates:

- creating structured `Document` objects with metadata
- loading text files and entire directories with LangChain loaders
- loading and parsing PDF documents
- splitting long documents into chunks for retrieval
- generating embeddings with local sentence-transformer models
- storing and querying embeddings in ChromaDB
- working with FAISS and Typesense as retrieval backends
- building prompt-driven RAG pipelines for answer generation
- adding source information, confidence signals, citations, summaries, and history
- implementing graph-based agentic orchestration with LangGraph

## Repository Structure

```text
.
├── agentic-rag/
│   └── 1-agenticrag.ipynb
├── notebook/
│   ├── document.ipynb
│   └── pdf_loader.ipynb
├── data/
│   ├── *.pdf
│   ├── text_files/
│   └── vector_store/
├── typesense.ipynb
├── books.jsonl
├── test.txt
├── pyproject.toml
├── requirements.txt
├── Traditional RAG.png
├── Agentic RAG.png
└── RAG FlowChart.png
```

## Notebook Guide

### `notebook/document.ipynb`

This notebook covers the fundamentals of working with LangChain documents.

Topics included:

- creating `Document` objects
- attaching metadata such as source and author
- generating simple text files for testing
- loading individual files with `TextLoader`
- loading multiple files with `DirectoryLoader`

This notebook is the starting point for understanding how raw source material becomes structured input for a retrieval pipeline.

### `notebook/pdf_loader.ipynb`

This is the main end-to-end RAG notebook in the repository and the most complete implementation.

It includes:

- loading PDF files recursively from the `data/` directory
- splitting documents with `RecursiveCharacterTextSplitter`
- generating embeddings with `sentence-transformers/all-MiniLM-L6-v2`
- persisting embeddings in ChromaDB under `data/vector_store/`
- retrieving top-k relevant chunks for a query
- generating answers with Groq
- extending the basic pipeline with richer outputs

The later sections of this notebook also introduce advanced RAG features such as:

- source attribution
- confidence scoring
- context previews
- optional full-context return
- summarization
- query history
- citation-style outputs

### `typesense.ipynb`

This notebook explores retrieval through Typesense in two different ways.

It demonstrates:

- connecting to a Typesense cluster
- creating a `books` collection
- importing data from `books.jsonl`
- running keyword-based search queries
- creating a LangChain-compatible Typesense vector store
- running similarity search and retriever-based queries

This notebook is useful if you want to compare specialized hosted search with local vector retrieval systems.

### `agentic-rag/1-agenticrag.ipynb`

This notebook demonstrates a lightweight agentic RAG workflow using LangGraph.

It includes:

- a typed shared state (`AgentState`)
- a node that decides whether retrieval is needed
- a retrieval node
- an answer-generation node
- conditional routing between nodes
- a compiled graph for end-to-end invocation

The implementation is intentionally simple, but it clearly shows the shift from fixed-pipeline RAG to graph-based, decision-driven orchestration.

## Data and Assets

### Demo text files

The repository includes small text documents for initial experiments:

- `data/text_files/python_intro.txt`
- `data/text_files/machine_learning.txt`

### PDF corpus

The `data/` directory contains the PDF corpus used for the ChromaDB-based traditional and advanced RAG pipeline. These documents are loaded, chunked, embedded, and stored for semantic retrieval.

### Typesense dataset

The file `books.jsonl` is used to create and populate the `books` collection in Typesense.

### Diagrams

The repository includes architecture images that summarize the main ideas behind each RAG variation:

- `RAG FlowChart.png`
- `Traditional RAG.png`
- `Agentic RAG.png`

## Tech Stack

### Core framework and orchestration

- `langchain`
- `langchain-core`
- `langchain-community`
- `langchain-text-splitters`
- `langgraph`

### LLM integrations

- `langchain-groq`
- `langchain-openai`

### Vector stores and retrieval backends

- `chromadb`
- `faiss-cpu`
- `typesense`

### Embeddings and runtime

- `sentence-transformers`
- `torch`
- `onnxruntime`

### Document processing

- `pypdf`
- `pymupdf`

### Utilities

- `python-dotenv`
- `numpy`
- `scikit-learn`
- `jupyter`
- `ipykernel`

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/HiteshM2107/RAG-Playground.git
cd "RAG-Playground"
```

### 2. Install dependencies

Using `uv`:

```bash
uv sync
```

Using `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Create a `.env` file

Add the API keys you need:

```env
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
TYPESENSE_API_KEY=your_typesense_api_key
```

Notes:

- `GROQ_API_KEY` is required for the LLM generation sections in `pdf_loader.ipynb`.
- `OPENAI_API_KEY` is used in the agentic notebook when `ChatOpenAI` is enabled.
- `TYPESENSE_API_KEY` is required for the Typesense notebook.
- local embeddings are also used in several parts of the project, so not every notebook step requires a paid API.

### 4. Launch Jupyter or VS Code

You can run the notebooks in Jupyter or VS Code. If you use VS Code, select the project kernel:

- `RAG-Playground`

## How to Run

The repository is best explored notebook by notebook.

### Foundational document loading

Run:

- `notebook/document.ipynb`

Use it to understand:

- `Document` structure
- metadata handling
- file loading
- directory loading

### Traditional and advanced RAG over PDFs

Run:

- `notebook/pdf_loader.ipynb`

Use it to:

- ingest PDFs
- split documents into chunks
- generate embeddings
- persist vectors in ChromaDB
- retrieve relevant chunks
- generate answers from retrieved context

### Typesense retrieval workflow

Run:

- `typesense.ipynb`

Use it to:

- create a Typesense collection
- import `books.jsonl`
- test search queries
- build a LangChain-compatible retriever

### Agentic RAG workflow

Run:

- `agentic-rag/1-agenticrag.ipynb`

Use it to:

- build a small vector store
- define a graph state
- route between retrieval and generation
- test a retrieval-aware workflow

## RAG Variants in This Repository

### Traditional RAG

Characteristics:

- fixed retrieval path
- single retrieval step
- retrieved context injected directly into the final prompt

Primary notebook:

- `notebook/pdf_loader.ipynb`

### Advanced RAG

Characteristics:

- richer outputs
- source metadata
- confidence estimates
- summaries
- answer history
- citation-style formatting

Primary notebook:

- `notebook/pdf_loader.ipynb`

### Agentic RAG

Characteristics:

- conditional retrieval
- graph-based execution
- explicit state transitions
- extensible decision logic

Primary notebook:

- `agentic-rag/1-agenticrag.ipynb`

### Typesense-Backed Retrieval

Characteristics:

- external search backend
- structured search over a book dataset
- retrieval through both the native client and LangChain integration

Primary notebook:

- `typesense.ipynb`

## Example End-to-End Flow

The strongest end-to-end workflow in the repo follows this pattern:

```text
PDFs
-> document loader
-> chunking
-> embeddings
-> ChromaDB
-> semantic retrieval
-> prompt construction
-> LLM generation
-> answer with sources
```

## Current Status

This repository is best described as a well-developed experimental and educational project.

At the moment:

- the core logic is implemented in notebooks rather than Python modules
- `main.py` is still a placeholder
- the project demonstrates multiple RAG architectures clearly
- the Chroma vector store is already persisted under `data/vector_store/`

## Suggested Next Steps

If you want to evolve the project further, strong follow-on improvements would be:

- move notebook logic into reusable modules under `src/`
- expose each RAG variant through a CLI or web interface
- add evaluation for retrieval quality and answer quality
- introduce reranking and hybrid search
- add tests for chunking, retrieval, and prompt construction
- package the project for easier reproducibility and sharing

## Troubleshooting

### Notebook kernel issues

If notebook cells fail to run:

- verify that the selected kernel is `RAG-Playground`
- restart the kernel
- reload the VS Code window if the notebook gets stuck connecting
- confirm dependencies were installed successfully

### OpenAI quota errors

If you see `429 insufficient_quota`:

- check OpenAI API billing and usage
- confirm your API key is active
- switch to local embeddings where appropriate if you do not want to use paid embeddings

### Groq model errors

If a Groq request fails:

- confirm the model name is still supported
- verify `GROQ_API_KEY` is present in `.env`

### Typesense errors

If Typesense requests fail:

- verify the host, port, and protocol
- confirm `TYPESENSE_API_KEY` is loaded
- handle collection-already-exists cases when rerunning notebook cells

## Summary

This project is a compact but broad survey of modern RAG patterns. It starts with document fundamentals, moves through traditional retrieval pipelines, extends into enhanced response generation, and ends with agentic orchestration and alternative retrieval backends. For anyone learning how RAG systems are built in practice, it provides a strong, incremental path from first principles to more advanced workflows.
