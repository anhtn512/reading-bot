import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
import faiss

from dotenv import load_dotenv
load_dotenv()

# Check for GOOGLE_API_KEY
if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY not set in environment variables")

# Configure Google GenAI embedding
Settings.embed_model = GoogleGenAIEmbedding(model_name="models/embedding-001")

# Load documents from directory
def load_documents(directory_path):
    if not os.path.exists(directory_path) or not any(f.endswith('.txt') for f in os.listdir(directory_path)):
        print(f"Directory {directory_path} is empty or contains no .txt files.")
        return []
    try:
        return SimpleDirectoryReader(input_dir=directory_path, required_exts=['.txt']).load_data()
    except Exception as e:
        print(f"Error loading documents: {e}")
        return []

# Create and save vector store
def create_and_save_index(directory_path, output_path):
    documents = load_documents(directory_path)
    if not documents:
        print("No documents found or directory is empty.")
        return
    
    # Initialize FAISS
    d = 768  # Vector size for embedding-001
    faiss_index = faiss.IndexFlatL2(d)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create index
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        store_nodes=True
    )
    
    # Save FAISS index
    faiss.write_index(faiss_index, os.path.join(output_path, "faiss_index.bin"))
    
    # Save storage context
    storage_context.persist(persist_dir=output_path)
    print(f"Vector store saved to {output_path}")

# Run embedding
if __name__ == "__main__":
    directory_path = "./documents"
    output_path = "./vector_store"
    os.makedirs(output_path, exist_ok=True)
    create_and_save_index(directory_path, output_path)