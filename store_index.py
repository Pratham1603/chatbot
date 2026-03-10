from dotenv import load_dotenv
import os
from src.helper import load_json_file, clean_documents, download_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import time

# 1. Setup & Config
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["HUGGINGFACE_API_KEY"] = HUGGINGFACE_API_KEY

# Set this to True if you want to delete the existing data and re-upload
FORCE_RELOAD = False                     
index_name = "mokshfit"

# 2. Initialize Pinecone & Embeddings
pc = Pinecone(api_key=PINECONE_API_KEY)
embeddings = download_embeddings()

# 3. Handle Index Creation/Deletion
if FORCE_RELOAD and pc.has_index(index_name):
    print(f"Force reload active. Deleting index: {index_name}...")
    pc.delete_index(index_name)
    time.sleep(2) # Give Pinecone a moment to register deletion

if not pc.has_index(index_name):
    print(f"Creating new index: {index_name}...")
    pc.create_index(
        name=index_name,
        dimension=768, 
        metric="cosine", 
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    # Wait for index to be ready
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

# 4. Check if Data Already Exists
index = pc.Index(index_name)
index_stats = index.describe_index_stats()
vector_count = index_stats['total_vector_count']

if vector_count > 0 and not FORCE_RELOAD:
    print(f"Index '{index_name}' already contains {vector_count} vectors. Skipping upload.")
else:
    print("Loading and processing documents...")
    extracted_docs = load_json_file("data/mokshfit.json")
    cleaned_docs = clean_documents(extracted_docs)

    # Generate Unique IDs
    doc_ids = []
    for i, doc in enumerate(cleaned_docs, start=1):
        title = doc.metadata.get("title", doc.metadata.get("handle", "Unknown"))
        clean_title = title.replace("-", " ")
        initials = "".join([word[0].upper() for word in clean_title.split() if word])
        unique_id = f"{initials}_{i}"
        doc_ids.append(unique_id)

    # 5. Batch Upsert to Pinecone
    print(f"Upserting {len(cleaned_docs)} documents in batches...")
    batch_size = 100 
    for i in range(0, len(cleaned_docs), batch_size):
        batch_docs = cleaned_docs[i : i + batch_size]
        batch_ids = doc_ids[i : i + batch_size]
        
        PineconeVectorStore.from_documents(
            documents=batch_docs,
            embedding=embeddings,
            index_name=index_name,
            ids=batch_ids 
        )
    print("Upload complete.")

# 6. Initialize Search Object
docsearch = PineconeVectorStore(index_name=index_name, embedding=embeddings)