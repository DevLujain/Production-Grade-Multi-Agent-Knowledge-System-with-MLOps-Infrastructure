"""
Synthesis Agent
Combines information from multiple sources into coherent answers with citations
"""
import os
from dotenv import load_dotenv
from groq import Groq
import re

load_dotenv()

class SynthesisAgent:
    def __init__(self, groq_api_key=None):
        """Initialize Synthesis Agent"""
        print("🧬 Initializing Synthesis Agent...\n")
        
        self.groq_client = Groq(api_key=groq_api_key)
        self.model = "llama-3.3-70b-versatile"
        
        self.system_prompt = """You are an expert at synthesizing information from multiple sources.

Given retrieved document chunks, create a comprehensive, well-structured answer that:
1. Combines relevant information from all sources
2. Uses chain-of-thought reasoning to connect ideas
3. Cites sources using [Source: source_name] notation
4. Explicitly states when information is uncertain or missing
5. Maintains factual accuracy without adding unsupported claims

Structure your answer clearly with logical flow and proper citations."""
        
        print("✅ Synthesis Agent ready!\n")
    
    def format_documents_for_synthesis(self, documents):
        """Format documents for the synthesis prompt"""
        formatted = "## RETRIEVED DOCUMENTS:\n\n"
        
        for i, doc in enumerate(documents, 1):
            formatted += f"[Document {i}] Source: {doc.get('source', 'unknown')}\n"
            formatted += f"Content: {doc['content'][:400]}...\n\n"
        
        return formatted
    
    def synthesize(self, query, documents):
        """Synthesize answer from multiple documents"""
        print(f"🧬 Synthesizing answer from {len(documents)} documents...")
        
        # Format documents
        formatted_docs = self.format_documents_for_synthesis(documents)
        
        # Create synthesis prompt
        prompt = f"""{formatted_docs}

QUESTION: {query}

Please synthesize a comprehensive answer based on these documents. Use chain-of-thought reasoning and cite your sources."""
        
        try:
            response = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model,
                temperature=0.7,
                max_tokens=1500
            )
            
            answer = response.choices[0].message.content.strip()
            print("✅ Synthesis complete!\n")
            
            return answer
        
        except Exception as e:
            print(f"❌ Synthesis error: {e}\n")
            return f"Error generating answer: {e}"
    
    def extract_citations(self, answer):
        """Extract citations from synthesized answer"""
        # Find all [Source: ...] patterns
        citations = re.findall(r'\[Source: ([^\]]+)\]', answer)
        return list(set(citations))  # Unique citations


# Test the agent
if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    
    agent = SynthesisAgent(groq_api_key=api_key)
    
    # Test documents
    test_docs = [
        {
            'source': 'fastapi.md',
            'content': 'FastAPI is a modern web framework for building APIs with Python. It uses standard Python type hints and is built on top of Starlette for the web parts and Pydantic for the data validation parts.'
        },
        {
            'source': 'python_docs.md',
            'content': 'Python is a high-level, general-purpose programming language. It emphasizes code readability with the use of significant whitespace.'
        }
    ]
    
    query = "What is FastAPI and how is it related to Python?"
    
    print("=" * 70)
    print("🧬 SYNTHESIS AGENT TEST")
    print("=" * 70 + "\n")
    
    answer = agent.synthesize(query, test_docs)
    
    print("SYNTHESIZED ANSWER:")
    print("-" * 70)
    print(answer)
    print("-" * 70 + "\n")
    
    citations = agent.extract_citations(answer)
    print(f"Citations found: {citations}")
