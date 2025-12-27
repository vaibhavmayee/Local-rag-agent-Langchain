# Local RAG Agent with LangChain

A **100% local**, beginner-friendly RAG (Retrieval-Augmented Generation) system using:
- **Ollama** (for LLM and embeddings)
- **LangChain ≥0.2.10** (for modular RAG pipeline)
- **ChromaDB** (for local vector storage)
- Supports **`.txt` and `.pdf`** files

> 🔒 No internet required after setup  
> 💡 Ideal for learning how RAG works under the hood

---
## Author : Vaibhav Vasant Mayee

## 🛠️ Setup Instructions

### 1. Install Ollama
Ollama runs your local LLM and embedding models.

- **macOS**: [Download from ollama.com](https://ollama.com/download/Ollama-darwin.zip)  
- **Windows/Linux**: See [Ollama Install Guide](https://ollama.com/download)

After installing, **start the Ollama app** (it runs in the background).

### 2. Pull Required Models
Open **Terminal** and run:

```bash
# Embedding model (converts text → vectors)
ollama pull nomic-embed-text:v1.5

# LLM for answering questions
ollama pull gemma3:4b

💡 These models are free, open, and run locally.
⏱️ First pull may take 5–10 minutes (downloads ~2–4 GB).

3. Install Python Dependencies
In your project folder, run:

bash
1
pip install langchain==0.2.10 langchain-community==0.2.10 chromadb pypdf ollama

✅ This avoids dependency conflicts.
🐍 Requires Python 3.9+.

📁 Folder Setup
Create a data/ folder and add your documents:

bash
123
mkdir data# Then copy your files into ./data/# Supported: .txt and .pdf (text-based only)

Example:

123456
Local-rag-agent-Langchain/├── rag_agent.py├── data/│   ├── notes.txt│   └── manual.pdf└── ...

⚠️ Scanned/image PDFs won’t work — this tool only reads text-based PDFs.

▶️ How to Use
Step 1: Ingest Your Documents (Run Once)
bash
1
python rag_agent.py --mode embedding --data_path ./data

What this does:

Reads all .txt and .pdf files in ./data/
Splits text into chunks (800 chars with 200-char overlap)
Generates embeddings using nomic-embed-text:v1.5
Saves everything to ./chroma_db (local folder)
✅ Success message: [SUCCESS] Ingested X chunks.

🔁 Run this again whenever you add new files to ./data/.

Step 2: Ask Questions
bash
1
python rag_agent.py --mode qa --query "What is the main idea of the document?"

What this does:

Loads your saved data from ./chroma_db
Finds top 3 most relevant text chunks
Sends them + your question to gemma3:4b
Prints an answer based only on your documents
✅ If the answer isn’t in your data, it replies:
This info is not available with me right now.

🧪 Test It (Quick Demo)
Create a test file:
bash
1
echo "RAG stands for Retrieval-Augmented Generation. It combines retrieval from a knowledge base with language model generation." > data/test.txt

Ingest:
bash
1
python rag_agent.py --mode embedding --data_path ./data

Ask:
bash
1
python rag_agent.py --mode qa --query "What does RAG stand for?"

✅ Expected output:
[ANSWER] RAG stands for Retrieval-Augmented Generation.