# peek.py
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    collection_name="local_docs",
    embedding_function=embeddings
)

data = vectorstore.get()
print(f"Total chunks: {len(data['ids'])}\n")

for i in range(min(3, len(data["ids"]))):
    print(f"📄 Chunk {i+1}")
    print(f"   Source: {data['metadatas'][i].get('source', 'N/A')}")
    print(f"   Text: {data['documents'][i][:200]}...\n")