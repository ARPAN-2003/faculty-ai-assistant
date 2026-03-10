from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load embedding model
emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Connect to ChromaDB
vectordb = Chroma(
    persist_directory="policy_db",
    embedding_function=emb
)

collection = vectordb._collection

print("Total documents stored:", collection.count())
print()

docs = collection.get()

for i, doc in enumerate(docs["documents"]):
    print(f"Document {i+1}:")
    print(doc)
    print("-" * 50)