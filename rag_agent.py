"""
Author : Vaibhav Vasant Mayee
Local RAG Agent with LangChain (v0.2.10+)
Supports .txt and .pdf | Uses Ollama + ChromaDB

HOW TO RUN:
1. Ingest mode: 
   python rag_agent.py --mode embedding --data_path ./data
   → Reads all .txt/.pdf in ./data, stores embeddings in ./chroma_db

2. QA mode:
   python rag_agent.py --mode qa --query "What is RAG?"
   → Retrieves relevant text from ./chroma_db, generates answer using LLM
"""

import os              # For file/folder operations (e.g., check if path exists)
import sys             # For exiting the script on error
import argparse        # For parsing command-line arguments (--mode, --data_path, etc.)
import glob            # For finding files matching a pattern (e.g., "*.txt")

# ✅ LANGCHAIN IMPORTS (v0.2.10+)
# These are verified to work together without import errors.
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Splits long text into chunks
from langchain_community.document_loaders import TextLoader, PyPDFLoader  # Load .txt and .pdf files
from langchain_community.embeddings import OllamaEmbeddings         # Generate embeddings using Ollama
from langchain_chroma import Chroma                                 # Store/retrieve vectors in ChromaDB
from langchain_community.llms import Ollama                        # Use local LLM via Ollama
from langchain_core.prompts import ChatPromptTemplate             # Format prompts for the LLM
from langchain_core.runnables import RunnablePassthrough          # Pass input through unchanged in chains
from langchain_core.output_parsers import StrOutputParser         # Parse LLM output as plain string

# === GLOBAL CONFIGURATION ===
# These values control model behavior and storage. Change them to customize.
CHROMA_DB_PATH = "./chroma_db"    # Folder where vector database is saved on disk.
                                  # → Creates ./chroma_db if it doesn't exist.
                                  # → Must match in BOTH embedding and QA modes.
EMBED_MODEL = "nomic-embed-text:v1.5"  # Embedding model name in Ollama.
                                       # → Must be pulled via: ollama pull nomic-embed-text:v1.5
                                       # → Determines how text is converted to vectors.
LLM_MODEL = "gemma3:4b"           # LLM for generating answers.
                                  # → Must be pulled via: ollama pull gemma3:4b
                                  # → Smaller models (e.g., 4b) run faster on limited hardware.
COLLECTION_NAME = "local_docs"    # Name of the document collection inside ChromaDB.
                                  # → Used to organize data; must be consistent across runs.

def ingest_documents(data_path: str):
    """
    INGEST MODE FUNCTION
    Reads all .txt and .pdf files in data_path, splits them into chunks,
    generates embeddings, and stores everything in ChromaDB.

    PARAMETER:
    - data_path (str): Path to folder containing .txt/.pdf files.
      → Example: "./data" (must be a real folder with files)

    WHAT HAPPENS:
    1. Finds all .txt and .pdf files in data_path
    2. Loads text content from each file
    3. Splits long documents into smaller chunks
    4. Converts chunks to numerical vectors (embeddings)
    5. Saves vectors + original text to ./chroma_db
    """
    print(f"[INFO] Ingesting documents from: {data_path}")
    
    # Find all .txt files in data_path (e.g., ["./data/notes.txt", "./data/readme.txt"])
    txt_files = glob.glob(os.path.join(data_path, "*.txt"))
    # Find all .pdf files in data_path (e.g., ["./data/guide.pdf"])
    pdf_files = glob.glob(os.path.join(data_path, "*.pdf"))
    all_files = txt_files + pdf_files  # Combine both lists

    # Error if no files found
    if not all_files:
        print(f"[ERROR] No .txt or .pdf files in {data_path}")
        return  # Exit function early

    documents = []  # Will hold all loaded document objects

    # --- LOAD .TXT FILES ---
    for fp in txt_files:
        try:
            # TextLoader reads plain text files.
            # encoding='utf-8': Ensures special characters (e.g., é, ü) are handled correctly.
            loader = TextLoader(fp, encoding='utf-8')
            # loader.load() returns a list of Document objects (even for 1 file)
            docs = loader.load()
            documents.extend(docs)  # Add to master list
        except Exception as e:
            print(f"[SKIP TXT] {fp}: {e}")  # Skip corrupted files

    # --- LOAD .PDF FILES ---
    for fp in pdf_files:
        try:
            # PyPDFLoader extracts text from PDF pages.
            # Note: Only works on text-based PDFs (not scanned images).
            loader = PyPDFLoader(fp)
            docs = loader.load()  # Returns one Document per PDF page
            documents.extend(docs)
        except Exception as e:
            print(f"[SKIP PDF] {fp}: {e}")  # Skip if PDF is encrypted/corrupted

    # Safety check: exit if no documents were loaded
    if not documents:
        print("[WARNING] No documents loaded.")
        return

    # --- SPLIT DOCUMENTS INTO CHUNKS ---
    # Why split? LLMs have token limits; small chunks fit better and improve retrieval.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,    # Max characters per chunk (including spaces).
                          # → Smaller = more precise retrieval, but more chunks.
                          # → Larger = fewer chunks, but may include irrelevant info.
        chunk_overlap=200  # Overlap between chunks (in characters).
                          # → Preserves context across boundaries (e.g., sentences split mid-way).
    )
    # split_documents() returns a list of smaller Document objects
    splits = text_splitter.split_documents(documents)

    # --- GENERATE EMBEDDINGS AND STORE IN CHROMADB ---
    # OllamaEmbeddings uses your local Ollama server to convert text → vectors.
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    
    # Chroma.from_documents() does 4 things:
    # 1. Generates embeddings for each chunk using `embeddings`
    # 2. Stores vectors + original text + metadata in memory
    # 3. Persists everything to disk at CHROMA_DB_PATH
    # 4. Names the collection COLLECTION_NAME
    vectorstore = Chroma.from_documents(
        documents=splits,             # List of Document objects to store
        embedding=embeddings,         # Embedding function to use
        persist_directory=CHROMA_DB_PATH,  # Where to save on disk
        collection_name=COLLECTION_NAME    # Name of this dataset
    )
    
    # Force-save to disk (in case of unexpected shutdown)
    vectorstore.persist()
    print(f"[SUCCESS] Ingested {len(splits)} chunks.")

def answer_question(query: str):
    """
    QA MODE FUNCTION
    Takes a user question, retrieves relevant context from ChromaDB,
    and uses the LLM to generate an answer.

    PARAMETER:
    - query (str): The user's question (e.g., "What is RAG?")
    
    WHAT HAPPENS:
    1. Loads existing ChromaDB from ./chroma_db
    2. Converts query to embedding
    3. Finds top 3 most similar text chunks
    4. Sends chunks + query to LLM
    5. Prints the LLM's answer
    """
    print(f"[INFO] Query: {query}")
    
    # Reuse the same embedding model used during ingestion
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    
    # Load existing ChromaDB from disk (must match CHROMA_DB_PATH and COLLECTION_NAME)
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,    # Where DB is saved
        collection_name=COLLECTION_NAME,     # Which collection to load
        embedding_function=embeddings        # How to embed new queries
    )
    
    # Create a "retriever" that finds similar documents
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}  # Return top 3 most relevant chunks.
                               # → Higher k = more context, but slower & noisier.
                               # → Lower k = faster, but may miss key info.
    )
    
    # Load the local LLM (must match LLM_MODEL)
    llm = Ollama(model=LLM_MODEL)

    # Define prompt template
    # {context} = retrieved text chunks
    # {question} = user's query
    # → Forces LLM to use ONLY provided context (reduces hallucination)
    prompt = ChatPromptTemplate.from_template(
        """Use ONLY the context below. If unsure, say:
        "This info is not available with me right now."

        Context: {context}
        Question: {question}
        Answer:"""
    )
    
    # Helper function: Convert list of Document objects → single string
    def format_docs(docs):
        # doc.page_content = actual text of the chunk
        return "\n\n".join(doc.page_content for doc in docs)

    # --- BUILD RAG CHAIN (LCEL SYNTAX) ---
    # This defines the data flow: query → retrieve → format → prompt → LLM → output
    rag_chain = (
        # Input: {"question": "What is RAG?"}
        {
            "context": retriever | format_docs,  # Retrieve docs → format as string
            "question": RunnablePassthrough()    # Pass query through unchanged
        }
        | prompt      # Insert context + question into template
        | llm         # Send to LLM
        | StrOutputParser()  # Convert LLM response to plain string
    )
    
    # Run the chain
    try:
        answer = rag_chain.invoke(query)  # query is passed as "question"
        print("[ANSWER]", answer.strip())
    except Exception as e:
        # Fallback if LLM fails (e.g., Ollama not running)
        print("[ANSWER] This info is not available with me right now, or the question is irrelevant.")

# === COMMAND-LINE INTERFACE ===
if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser()
    
    # --mode: Required. Must be "embedding" or "qa"
    parser.add_argument("--mode", choices=["embedding", "qa"], required=True)
    
    # --data_path: Required only for embedding mode
    # → Should point to a folder (e.g., "./data")
    parser.add_argument("--data_path", help="Folder with .txt/.pdf files")
    
    # --query: Required only for qa mode
    # → Should be a string question (e.g., "Explain RAG")
    parser.add_argument("--query", help="Question for QA mode")

    # Parse command-line arguments
    args = parser.parse_args()

    # --- EMBEDDING MODE ---
    if args.mode == "embedding":
        # Validate that --data_path was provided and is a real folder
        if not args.data_path or not os.path.isdir(args.data_path):
            print("[ERROR] --data_path must be a valid folder")
            sys.exit(1)  # Exit with error code
        # Call ingest function with user-provided path
        ingest_documents(args.data_path)

    # --- QA MODE ---
    elif args.mode == "qa":
        # Validate that --query was provided
        if not args.query:
            print("[ERROR] --query is required in QA mode")
            sys.exit(1)
        # Call QA function with user's question
        answer_question(args.query)
