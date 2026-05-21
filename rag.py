import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
import chromadb

# Configure the LLM and embedding model
Settings.llm = OpenAI(model="gpt-4o", temperature=0.1)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Set up ChromaDB as our local vector store
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("dutchbros")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Load documents from the data folder
print("Loading documents...")
documents = SimpleDirectoryReader("data").load_data()
print(f"Loaded {len(documents)} document(s)")

# Build the index - this chunks, embeds, and stores everything
print("Building index...")
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)
print("Index built successfully!")

# Create a query engine
query_engine = index.as_query_engine(similarity_top_k=3)

# Interactive query loop
print("\n--- Dutch Bros Knowledge Base ---")
print("Ask anything about Dutch Bros. Type 'quit' to exit.\n")

while True:
    question = input("Your question: ")
    if question.lower() == "quit":
        break

    response = query_engine.query(question)
    print(f"\nAnswer: {response}\n")
    print("-" * 50 + "\n")
