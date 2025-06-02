import os
import asyncio
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import StorageContext
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
import faiss
import platform
from dotenv import load_dotenv
load_dotenv()

# Configure Gemini API key from environment
if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY not set in environment variables")

# Configure Gemini with Vietnamese output
Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001")
Settings.llm = Gemini(model_name="models/gemini-1.5-flash")

# Load vector store
def load_index(persist_dir):
    if not os.path.exists(persist_dir):
        raise ValueError(f"Persist directory {persist_dir} does not exist.")
    try:
        # Load FAISS vector store
        vector_store = FaissVectorStore.from_persist_dir(persist_dir)
        # Load storage context with vector store
        storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=persist_dir)
        # Rebuild index from vector store
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context
        )
        return index
    except Exception as e:
        print(f"Error loading vector store: {e}")
        raise

# Query index with Vietnamese response
def query_index(index, query):
    query_engine = index.as_query_engine(
        response_mode="compact",
        system_prompt="Trả lời bằng tiếng Việt, chỉ dựa vào nội dung tài liệu."
    )
    return str(query_engine.query(query))

# Main function
async def main():
    persist_dir = "./vector_store"
    
    # Load index
    print("Đang tải vector store...")
    try:
        index = load_index(persist_dir)
    except Exception as e:
        print(f"Lỗi khi tải vector store: {e}")
        return
    
    # Query loop
    while True:
        query = input("\nNhập câu hỏi của bạn (hoặc 'exit' để thoát): ")
        if query.lower() == 'exit':
            break
        
        # Get response from documents
        response = query_index(index, query)
        print(f"\nTrả lời từ tài liệu: {response}")

# Run application
if __name__ == "__main__":
    asyncio.run(main())