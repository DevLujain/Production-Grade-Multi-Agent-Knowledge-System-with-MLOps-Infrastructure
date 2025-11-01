"""
Query Understanding Agent
Reformulates vague/ambiguous queries into precise search queries
"""
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

class QueryUnderstandingAgent:
    def __init__(self, groq_api_key=None):
        """Initialize Query Understanding Agent"""
        print("🧠 Initializing Query Understanding Agent...\n")
        
        self.groq_client = Groq(api_key=groq_api_key)
        self.model = "llama-3.3-70b-versatile"
        
        self.system_prompt = """You are a query reformulation expert. Your task is to take vague or ambiguous user queries and reformulate them into precise, specific search queries that will retrieve the most relevant information.

Guidelines:
1. Expand acronyms (e.g., "API" → "Application Programming Interface")
2. Add context when needed
3. Break down complex multi-part questions into clear components
4. Make implicit requirements explicit
5. Keep reformulated query concise but comprehensive

Examples:
- Vague: "How do I make an API?"
  Reformulated: "How do I create a REST API endpoint using FastAPI?"

- Vague: "What about leave?"
  Reformulated: "What is the employee leave policy and how do I request leave?"

- Vague: "Remote work stuff"
  Reformulated: "What are the remote work policies and guidelines?"

Return ONLY the reformulated query, nothing else."""
    
    def reformulate_query(self, user_query):
        """Reformulate a vague query into a precise search query"""
        print(f"📝 Original query: '{user_query}'")
        
        try:
            response = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_query
                    }
                ],
                model=self.model,
                temperature=0.3,  # Lower temp for consistency
                max_tokens=200
            )
            
            reformulated = response.choices[0].message.content.strip()
            print(f"✨ Reformulated: '{reformulated}'\n")
            
            return reformulated
        
        except Exception as e:
            print(f"❌ Error reformulating query: {e}\n")
            return user_query  # Return original if error


# Test the agent
if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    
    agent = QueryUnderstandingAgent(groq_api_key=api_key)
    
    test_queries = [
        "How do I make an API?",
        "What about leave?",
        "Remote work stuff",
        "How to get docs?",
        "Tell me about policies"
    ]
    
    print("=" * 70)
    print("🧠 QUERY UNDERSTANDING AGENT TEST")
    print("=" * 70 + "\n")
    
    for query in test_queries:
        reformulated = agent.reformulate_query(query)
        print("-" * 70 + "\n")
