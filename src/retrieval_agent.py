"""
Multi-Source Retrieval Agent
Intelligently decides which sources to query based on query type
"""
import os
from dotenv import load_dotenv
from groq import Groq
from hybrid_search import HybridSearch
from sentence_transformers import SentenceTransformer

load_dotenv()

class RetrievalAgent:
    def __init__(self, chromadb_collection, groq_api_key=None):
        """Initialize Retrieval Agent"""
        print("🔍 Initializing Multi-Source Retrieval Agent...\n")
        
        self.groq_client = Groq(api_key=groq_api_key)
        self.model_name = "llama-3.3-70b-versatile"
        self.collection = chromadb_collection
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Initialize retrieval sources
        all_docs = self._get_all_documents()
        self.hybrid_search = HybridSearch(all_docs)
        
        self.classification_prompt = """Analyze this query and classify it:

QUERY: "{query}"

Determine:
1. Query Type: factual, conceptual, procedural, comparative
2. Information Need: general knowledge, specific details, step-by-step guide, comparison
3. Search Strategy: broad (many results), narrow (specific results), mixed

Respond in this format ONLY:
TYPE: [type]
NEED: [need]
STRATEGY: [strategy]"""
        
        print("✅ Retrieval Agent ready!\n")
    
    def _get_all_documents(self):
        """Get all documents from ChromaDB collection"""
        try:
            results = self.collection.get()
            docs = []
            for i, doc in enumerate(results['documents']):
                docs.append(doc)
            return docs
        except:
            return []
    
    def classify_query(self, query):
        """Use LLM to classify query for optimal retrieval strategy"""
        print(f"📊 Classifying query: '{query}'")
        
        try:
            response = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": self.classification_prompt.format(query=query)
                    }
                ],
                model=self.model_name,
                temperature=0.3,
                max_tokens=100
            )
            
            classification = response.choices[0].message.content.strip()
            print(f"✅ Classification:\n{classification}\n")
            
            return classification
        
        except Exception as e:
            print(f"❌ Classification error: {e}\n")
            return "TYPE: mixed\nNEED: general\nSTRATEGY: mixed"
    
    def vector_search(self, query, top_k=5):
        """Search using vector embeddings (semantic similarity)"""
        print(f"  📌 Performing vector search...")
        
        try:
            query_embedding = self.embedding_model.encode([query])[0]
            
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            
            vector_results = []
            if results and results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    vector_results.append({
                        'index': i,
                        'content': doc,
                        'source': results['metadatas'][0][i]['source_file'],
                        'score': 1 - results['distances'][0][i],
                        'method': 'vector_search'
                    })
            
            print(f"     ✓ Found {len(vector_results)} results via vector search")
            return vector_results
        
        except Exception as e:
            print(f"     ✗ Vector search error: {e}")
            return []
    
    def bm25_search(self, query, top_k=5):
        """Search using BM25 (keyword matching)"""
        print(f"  📌 Performing BM25 search...")
        
        try:
            bm25_results = self.hybrid_search.bm25_search(query, top_k)
            
            # Get all documents to find sources
            all_results = self.collection.get()
            doc_to_source = {}
            if all_results and all_results['metadatas']:
                for i, metadata in enumerate(all_results['metadatas']):
                    if i < len(all_results['documents']):
                        doc_text = all_results['documents'][i][:50]  # First 50 chars as key
                        doc_to_source[doc_text] = metadata.get('source_file', 'unknown')
            
            formatted_results = []
            for result in bm25_results:
                # Normalize BM25 score (typically 0-100, divide by 100)
                normalized_score = min(result['score'] / 100.0, 1.0)
                
                # Find source
                doc_preview = result['content'][:50]
                source = 'unknown'
                for key, val in doc_to_source.items():
                    if key in result['content']:
                        source = val
                        break
                
                formatted_results.append({
                    'index': result['index'],
                    'content': result['content'],
                    'source': source,
                    'score': normalized_score,
                    'method': 'bm25_search'
                })
            
            print(f"     ✓ Found {len(formatted_results)} results via BM25")
            return formatted_results
        
        except Exception as e:
            print(f"     ✗ BM25 search error: {e}")
            return []
    
    def retrieve(self, query, top_k=5):
        """
        Main retrieval method: intelligently combines multiple sources
        """
        print(f"\n🔍 RETRIEVING FOR QUERY: '{query}'")
        print("-" * 70)
        
        # Step 1: Classify query
        classification = self.classify_query(query)
        
        # Step 2: Decide which sources to use
        use_vector = True  # Always use vector
        use_bm25 = True    # Always use BM25
        
        all_results = []
        
        print(f"🔎 Searching sources:")
        
        # Step 3: Search vector database
        if use_vector:
            vector_results = self.vector_search(query, top_k)
            all_results.extend(vector_results)
        
        # Step 4: Search BM25
        if use_bm25:
            bm25_results = self.bm25_search(query, top_k)
            all_results.extend(bm25_results)
        
        # Step 5: Deduplicate and rank
        seen = set()
        unique_results = []
        
        for result in all_results:
            content_hash = hash(result['content'][:100])
            if content_hash not in seen:
                seen.add(content_hash)
                unique_results.append(result)
        
        # Sort by score (descending)
        unique_results.sort(key=lambda x: x['score'], reverse=True)
        final_results = unique_results[:top_k]
        
        print(f"\n✅ Retrieved {len(final_results)} unique documents")
        print("-" * 70 + "\n")
        
        return final_results


# Test the agent
if __name__ == "__main__":
    import chromadb
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    
    # Connect to ChromaDB
    client = chromadb.PersistentClient(path="data/vectordb")
    collection = client.get_collection(name="documents")
    
    # Initialize agent
    agent = RetrievalAgent(collection, groq_api_key=api_key)
    
    # Test queries
    test_queries = [
        "How do I create a FastAPI endpoint?",
        "What is the leave policy?",
        "Remote work guidelines"
    ]
    
    print("=" * 70)
    print("🔍 MULTI-SOURCE RETRIEVAL AGENT TEST")
    print("=" * 70)
    
    for query in test_queries:
        results = agent.retrieve(query, top_k=3)
        print(f"Results for '{query}':")
        for i, result in enumerate(results, 1):
            print(f"  {i}. [{result['method']}] Score: {result['score']:.2f}")
        print()
