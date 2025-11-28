import json
import chromadb
from chromadb.utils import embedding_functions

# --- CONFIGURATION ---
JSON_PATH = "data/video_index.json"
DB_PATH = "data/video_db"  # Where the vector memory will be saved
COLLECTION_NAME = "video_memory"

def get_chroma_client():
    # This saves the database to your disk so we don't have to rebuild it every time
    return chromadb.PersistentClient(path=DB_PATH)

def index_data():
    """Reads the JSON and puts it into the Vector Database"""
    print("🧠 Loading ChromaDB...")
    client = get_chroma_client()
    
    # Use a high-quality open source embedding model
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    # Create (or reset) the collection
    try:
        client.delete_collection(name=COLLECTION_NAME) # Reset if exists
    except:
        
        pass
    
    collection = client.create_collection(name=COLLECTION_NAME, embedding_function=ef)
    
    # Load the Day 1 data
    with open(JSON_PATH, "r") as f:
        data = json.load(f)
        
    print(f"   📂 Found {len(data)} items to memorize.")
    
    # Prepare data for ChromaDB
    documents = []
    metadatas = []
    ids = []
    
    for idx, item in enumerate(data):
        documents.append(item['content'])
        metadatas.append({
            "start": item['start'], 
            "end": item['end'],
            "type": item['type']
        })
        ids.append(f"id_{idx}")
    
    # Batch add (Chroma handles batches well)
    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print("✅ Knowledge stored in Vector Database!")
    return collection

def search_video(query, top_k=3):
    """Searches the database for the query"""
    client = get_chroma_client()
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)
    
    results = collection.query(query_texts=[query], n_results=top_k)
    
    # Format results nicely
    formatted_results = []
    for i in range(len(results['documents'][0])):
        formatted_results.append({
            "content": results['documents'][0][i],
            "start": results['metadatas'][0][i]['start'],
            "type": results['metadatas'][0][i]['type']
        })
    
    return formatted_results

# --- TEST AREA ---
if __name__ == "__main__":
    # 1. Run Indexing (Only need to do this once per video)
    index_data()
    
    # 2. Test Search
    while True:
        user_query = input("\n🔎 Ask something about the video (or 'q' to quit): ")
        if user_query.lower() == 'q': break
        
        answers = search_video(user_query)
        print(f"\nTop Match ({answers[0]['type']}):")
        print(f"Timestamp: {int(answers[0]['start'])}s")
        print(f"Context: {answers[0]['content']}")