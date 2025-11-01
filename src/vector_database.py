import json
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path

class VectorDatabase:
    def __init__(self, db_path="data/vectordb", model_name="all-MiniLM-L6-v2"):
        """
        Initialize vector database
        
        Args:
            db_path: Path to store vector database
            model_name: Sentence transformer model to use
        """
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Load embedding model
        print(f"📦 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("✅ Model loaded!")
    
    def load_documents(self, json_path):
        """Load documents from JSON file"""
        print(f"\n📂 Loading documents from {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        print(f"✅ Loaded {len(documents)} documents")
        return documents
    
    def create_embeddings(self, documents):
        """Create embeddings for all documents"""
        print(f"\n🔄 Creating embeddings for {len(documents)} documents...")
        
        texts = [doc['content'] for doc in documents]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        print(f"✅ Created {len(embeddings)} embeddings")
        return embeddings
    
    def store_documents(self, documents, embeddings):
        """Store documents and embeddings in ChromaDB"""
        print(f"\n💾 Storing documents in ChromaDB...")
        
        # Prepare data for ChromaDB
        ids = [doc['doc_id'] for doc in documents]
        texts = [doc['content'] for doc in documents]
        metadatas = [
            {
                'title': doc['title'],
                'word_count': str(doc['word_count']),
                'source_file': doc['source_file']
            }
            for doc in documents
        ]
        
        # Convert embeddings to list format
        embeddings_list = [emb.tolist() for emb in embeddings]
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            documents=texts,
            metadatas=metadatas
        )
        
        print(f"✅ Stored {len(documents)} documents in ChromaDB")
    
    def search(self, query, top_k=5):
        """Search for similar documents"""
        print(f"\n🔍 Searching for: '{query}'")
        
        # Create embedding for query
        query_embedding = self.model.encode([query])[0]
        
        # Search in collection
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        return results
    
    def display_results(self, results):
        """Display search results in readable format"""
        if not results or not results['documents'] or len(results['documents'][0]) == 0:
            print("❌ No results found")
            return
        
        print(f"\n✅ Found {len(results['documents'][0])} results:\n")
        
        for i, (doc, distance, metadata) in enumerate(
            zip(
                results['documents'][0],
                results['distances'][0],
                results['metadatas'][0]
            )
        ):
            print(f"--- Result {i+1} ---")
            print(f"Title: {metadata['title']}")
            print(f"Source: {metadata['source_file']}")
            print(f"Similarity Score: {1 - distance:.3f}")
            print(f"Preview: {doc[:200]}...")
            print()


# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 VECTOR DATABASE SETUP")
    print("=" * 60)
    
    # Initialize vector database
    vdb = VectorDatabase()
    
    # Load documents
    documents = vdb.load_documents("data/processed/processed_documents.json")
    
    # Create embeddings
    embeddings = vdb.create_embeddings(documents)
    
    # Store in database
    vdb.store_documents(documents, embeddings)
    
    # Test search
    print("\n" + "=" * 60)
    print("🧪 TESTING SEARCH")
    print("=" * 60)
    
    test_queries = [
        "How do I create a FastAPI endpoint?",
        "What is employee leave policy?",
        "How do I work remotely?"
    ]
    
    for query in test_queries:
        results = vdb.search(query, top_k=3)
        vdb.display_results(results)
    
    print("\n" + "=" * 60)
    print("✅ VECTOR DATABASE SETUP COMPLETE!")
    print("=" * 60)
