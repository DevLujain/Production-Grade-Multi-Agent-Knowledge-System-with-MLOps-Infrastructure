"""
Agent Orchestrator
Connects all agents using LangGraph workflow
"""
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Dict

load_dotenv()

class AgentState(TypedDict):
    """State passed between agents"""
    original_query: str
    reformulated_query: str
    retrieved_documents: List[Dict]
    synthesized_answer: str
    validation_result: Dict
    final_answer: str
    metadata: Dict


class AgentOrchestrator:
    def __init__(self, rag_system):
        """Initialize orchestrator with RAG system"""
        print("🔗 Initializing Agent Orchestrator...\n")
        
        self.rag = rag_system
        self.workflow = self._build_workflow()
        
        print("✅ Agent Orchestrator ready!\n")
    
    def _build_workflow(self):
        """Build LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Define nodes
        workflow.add_node("query_understanding", self._query_understanding_node)
        workflow.add_node("retrieval", self._retrieval_node)
        workflow.add_node("synthesis", self._synthesis_node)
        workflow.add_node("validation", self._validation_node)
        workflow.add_node("finalize", self._finalize_node)
        
        # Define edges
        workflow.add_edge(START, "query_understanding")
        workflow.add_edge("query_understanding", "retrieval")
        workflow.add_edge("retrieval", "synthesis")
        workflow.add_edge("synthesis", "validation")
        workflow.add_edge("validation", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def _query_understanding_node(self, state: AgentState) -> AgentState:
        """Query Understanding Agent Node"""
        print("\n" + "=" * 70)
        print("🧠 AGENT 1: QUERY UNDERSTANDING")
        print("=" * 70)
        
        original_query = state["original_query"]
        reformulated_query = self.rag.query_agent.reformulate_query(original_query)
        
        state["reformulated_query"] = reformulated_query
        state["metadata"]["query_understanding_time"] = 0
        
        return state
    
    def _retrieval_node(self, state: AgentState) -> AgentState:
        """Multi-Source Retrieval Agent Node"""
        print("\n" + "=" * 70)
        print("🔍 AGENT 2: MULTI-SOURCE RETRIEVAL")
        print("=" * 70)
        
        reformulated_query = state["reformulated_query"]
        retrieved_results = self.rag.retrieval_agent.retrieve(reformulated_query, top_k=5)
        
        # Convert to document format
        documents = []
        for result in retrieved_results:
            documents.append({
                'content': result['content'],
                'source': result.get('source', 'unknown'),
                'score': result['score']
            })
        
        state["retrieved_documents"] = documents
        state["metadata"]["num_documents_retrieved"] = len(documents)
        
        return state
    
    def _synthesis_node(self, state: AgentState) -> AgentState:
        """Synthesis Agent Node"""
        print("\n" + "=" * 70)
        print("🧬 AGENT 3: SYNTHESIS")
        print("=" * 70)
        
        original_query = state["original_query"]
        documents = state["retrieved_documents"]
        
        synthesized_answer = self.rag.synthesis_agent.synthesize(
            original_query, 
            documents
        )
        
        state["synthesized_answer"] = synthesized_answer
        
        return state
    
    def _validation_node(self, state: AgentState) -> AgentState:
        """Validation Agent Node"""
        print("\n" + "=" * 70)
        print("✅ AGENT 4: VALIDATION")
        print("=" * 70)
        
        answer = state["synthesized_answer"]
        documents = state["retrieved_documents"]
        
        validation_result = self.rag.validation_agent.validate(answer, documents)
        
        state["validation_result"] = validation_result
        
        return state
    
    def _finalize_node(self, state: AgentState) -> AgentState:
        """Finalize and format response"""
        print("\n" + "=" * 70)
        print("📋 FINALIZATION")
        print("=" * 70 + "\n")
        
        state["final_answer"] = state["synthesized_answer"]
        
        return state
    
    def run(self, query: str) -> Dict:
        """Run complete agent orchestration workflow"""
        print("\n" + "=" * 80)
        print("🚀 MULTI-AGENT ORCHESTRATION WORKFLOW")
        print("=" * 80)
        print(f"\nINPUT QUERY: {query}\n")
        
        # Initialize state
        initial_state = {
            "original_query": query,
            "reformulated_query": "",
            "retrieved_documents": [],
            "synthesized_answer": "",
            "validation_result": {},
            "final_answer": "",
            "metadata": {}
        }
        
        # Run workflow
        final_state = self.workflow.invoke(initial_state)
        
        # Format and display results
        self._display_results(final_state)
        
        return final_state
    
    def _display_results(self, state: AgentState):
        """Display final results"""
        print("\n" + "=" * 80)
        print("🎯 FINAL RESULTS")
        print("=" * 80 + "\n")
        
        print("ORIGINAL QUERY:")
        print(f"  {state['original_query']}\n")
        
        print("REFORMULATED QUERY:")
        print(f"  {state['reformulated_query']}\n")
        
        print("ANSWER:")
        print("-" * 80)
        print(state['final_answer'])
        print("-" * 80 + "\n")
        
        validation = state['validation_result']
        print("VALIDATION:")
        print(f"  Status: {'✅ VALID' if validation['is_valid'] else '⚠️ NEEDS REVIEW'}")
        print(f"  Confidence: {validation['confidence']}%\n")
        
        print("SOURCES:")
        for i, doc in enumerate(state['retrieved_documents'], 1):
            print(f"  {i}. {doc['source']} (relevance: {doc['score']:.2%})")
        
        print("\n" + "=" * 80 + "\n")


# Test the orchestrator
if __name__ == "__main__":
    from rag_system import RAGSystem
    
    api_key = os.getenv("GROQ_API_KEY")
    
    # Initialize RAG system
    print("Initializing RAG System...")
    rag = RAGSystem(groq_api_key=api_key)
    
    # Initialize orchestrator
    orchestrator = AgentOrchestrator(rag)
    
    # Test queries
    test_queries = [
        "How do I create a FastAPI endpoint?",
        "What is the leave policy?",
        "Tell me about remote work"
    ]
    
    for query in test_queries:
        result = orchestrator.run(query)
        print("\n" + "=" * 80 + "\n")
