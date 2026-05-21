import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import chromadb

app = Flask(__name__)
CORS(app)

# Configure settings
Settings.llm = OpenAI(
    model="gpt-4o",
    temperature=0.1,
    system_prompt="""You are a helpful knowledge base assistant for Dutch Bros Coffee. 
    Answer questions based only on the information provided to you.
    If you don't have the answer in your knowledge base, say clearly: 
    'I don't have that information in my knowledge base. Please check with your manager or visit dutchbros.com.'
    Never make up information. Always be friendly and concise.""",
)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Initialize ChromaDB and index
print("Initializing knowledge base...")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("dutchbros")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)
query_engine = index.as_query_engine(similarity_top_k=3)
print("Knowledge base ready!")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    response = query_engine.query(question)
    return jsonify({"answer": str(response)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
