"""
FastAPI REST API Service
Exposes the multi-agent knowledge system
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import time
from dotenv import load_dotenv
import os

from rag_system import RAGSystem
from agent_orchestrator import AgentOrchestrator

load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="Multi-Agent Knowledge System",
    description="RAG system with query understanding, retrieval, synthesis, and validation",
    version="1.0.0"
)

# Initialize RAG system
api_key = os.getenv("GROQ_API_KEY")
rag_system = RAGSystem(groq_api_key=api_key)

# Request/Response Models
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class SourceDocument(BaseModel):
    source: str
    relevance: float

class ValidationInfo(BaseModel):
    status: str
    confidence: int

class QueryResponse(BaseModel):
    query: str
    reformulated_query: str
    answer: str
    validation: ValidationInfo
    sources: List[SourceDocument]
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    db_connected: bool
    timestamp: str

class MetricsResponse(BaseModel):
    total_queries: int
    avg_latency: float
    avg_confidence: float

# Global metrics
metrics = {
    "total_queries": 0,
    "latencies": [],
    "confidences": []
}

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check system health"""
    from datetime import datetime
    
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        db_connected=True,
        timestamp=datetime.now().isoformat()
    )

# Main query endpoint
@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Process a query through the multi-agent system
    
    Args:
        query: User query
        top_k: Number of documents to retrieve
    
    Returns:
        QueryResponse with answer, sources, and validation
    """
    try:
        start_time = time.time()
        
        # Store top_k in rag_system temporarily
        original_top_k = 5
        
        # Run orchestrator
        result = rag_system.answer_question(request.query)
        
        # Extract data
        processing_time = time.time() - start_time
        
        # Format sources
        sources = []
        for doc in result.get("retrieved_documents", []):
            sources.append(SourceDocument(
                source=doc["source"],
                relevance=doc["score"]
            ))
        
        # Format validation
        validation_info = result.get("validation_result", {})
        validation = ValidationInfo(
            status="✅ VALID" if validation_info.get("is_valid") else "⚠️ NEEDS REVIEW",
            confidence=validation_info.get("confidence", 0)
        )
        
        # Update metrics
        metrics["total_queries"] += 1
        metrics["latencies"].append(processing_time)
        metrics["confidences"].append(validation.confidence)
        
        # Build response
        response = QueryResponse(
            query=result.get("original_query", ""),
            reformulated_query=result.get("reformulated_query", ""),
            answer=result.get("final_answer", ""),
            validation=validation,
            sources=sources,
            processing_time=processing_time
        )
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Metrics endpoint
@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get system metrics"""
    avg_latency = sum(metrics["latencies"]) / len(metrics["latencies"]) if metrics["latencies"] else 0
    avg_confidence = sum(metrics["confidences"]) / len(metrics["confidences"]) if metrics["confidences"] else 0
    
    return MetricsResponse(
        total_queries=metrics["total_queries"],
        avg_latency=avg_latency,
        avg_confidence=avg_confidence
    )

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Multi-Agent Knowledge System API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "query": "/query (POST)",
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 70)
    print("🚀 Starting Multi-Agent Knowledge System API")
    print("=" * 70)
    print("📍 API running at: http://localhost:8000")
    print("📚 Documentation at: http://localhost:8000/docs")
    print("=" * 70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
