import json
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from .hybrid_search import HybridSearch
from query_agent import QueryUnderstandingAgent
from retrieval_agent import RetrievalAgent
from synthesis_agent import SynthesisAgent
from validation_agent import ValidationAgent


class RAGSystem:
    def __init__(self, db_path="data/vectordb", groq_api_key=None):
        """
        Initialize RAG System with Groq API
        """
        print("🔄 Initializing RAG System with Groq...\n")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name="documents")
        
        # Load embedding model
        print("📦 Loading embedding model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        print("✅ Model loaded!\n")
        
        # Initialize Groq client
        self.groq_client = Groq(api_key=groq_api_key)
        self.model_name = "llama-3.3-70b-versatile"  # Fast and good quality

# Initialize Hybrid Search
        print("🔀 Setting up hybrid search...")
        all_docs = [doc['content'] for doc in self.get_all_documents()]
        self.hybrid_search = HybridSearch(all_docs)
        print("✅ Hybrid search ready!\n")    
# Initialize Query Understanding Agent
        print("🧠 Setting up Query Understanding Agent...")
        self.query_agent = QueryUnderstandingAgent(groq_api_key=groq_api_key)
        print("✅ Query Agent ready!\n")
# Initialize Multi-Source Retrieval Agent
        print("🔍 Setting up Multi-Source Retrieval Agent...")
        self.retrieval_agent = RetrievalAgent(self.collection, groq_api_key=groq_api_key)
        print("✅ Retrieval Agent ready!\n")
# Initialize Synthesis Agent
        print("🧬 Setting up Synthesis Agent...")
        self.synthesis_agent = SynthesisAgent(groq_api_key=groq_api_key)
        print("✅ Synthesis Agent ready!\n")
# Initialize Validation Agent
        print("✅ Setting up Validation Agent...")
        self.validation_agent = ValidationAgent(groq_api_key=groq_api_key)
        print("✅ Validation Agent ready!\n")

    def retrieve_documents(self, query, top_k=5):
        """Retrieve relevant documents from vector database"""
        print(f"🔍 Retrieving documents for: '{query}'")
        
        # Create query embedding
        query_embedding = self.model.encode([query])[0]
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        # Format retrieved documents
        retrieved_docs = []
        if results and results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                retrieved_docs.append({
                    'content': doc,
                    'source': results['metadatas'][0][i]['source_file'],
                    'score': 1 - results['distances'][0][i]
                })
        
        print(f"✅ Retrieved {len(retrieved_docs)} documents\n")
        return retrieved_docs
    
    def format_context(self, documents):
        """Format retrieved documents as context for LLM"""
        context = "## RETRIEVED DOCUMENTS:\n\n"
        
        for i, doc in enumerate(documents, 1):
            context += f"[Document {i}] (Source: {doc['source']})\n"
            context += f"{doc['content'][:500]}...\n\n"
        
        return context
    
    def query_groq(self, prompt):
        """Send prompt to Groq API and get response"""
        print("🤖 Generating answer with Groq...\n")
        
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model_name,
                temperature=0.7,
                max_tokens=1500
            )
            
            return chat_completion.choices[0].message.content
        
        except Exception as e:
            return f"❌ Error with Groq API: {e}"
   
    def get_all_documents(self):
        """Get all documents from collection"""
        results = self.collection.get()
        docs = []
        for i, doc in enumerate(results['documents']):
            docs.append({
                'index': i,
                'content': doc,
                'source': results['metadatas'][i]['source_file'] if results['metadatas'] else 'unknown'
            })
        return docs

    def answer_question(self, query):
        """Use agent orchestrator for workflow"""
        if not hasattr(self, 'orchestrator'):
            from agent_orchestrator import AgentOrchestrator
            self.orchestrator = AgentOrchestrator(self)
        
        return self.orchestrator.run(query)       
       


# Main execution
if __name__ == "__main__":
    import os
    
    print("=" * 70)
    print("🚀 RAG SYSTEM WITH GROQ API")
    print("=" * 70 + "\n")
    
    # Get API key from environment or ask user
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("❌ Error: GROQ_API_KEY environment variable not set")
        print("\nTo set it, run:")
        print('  export GROQ_API_KEY="your_key_here"')
        print("\nThen run this script again")
        exit(1)
    
    # Initialize RAG system
    rag = RAGSystem(groq_api_key=groq_api_key)
    
    # Test questions
    test_questions = [
        "How do I create a FastAPI endpoint?",
        "What is the employee leave policy?",
        "How can I work remotely?"
    ]
    
    # Answer each question
    for question in test_questions:
        rag.answer_question(question)
        print("\n")
