"""
Hybrid Search: Combines Vector Search + BM25 Sparse Retrieval
"""
from rank_bm25 import BM25Okapi
import numpy as np

class HybridSearch:
    def __init__(self, documents):
        """
        Initialize BM25 index
        documents: list of document texts
        """
        print("📚 Building BM25 index...")
        
        # Tokenize documents
        self.tokenized_docs = [doc.lower().split() for doc in documents]
        self.documents = documents
        
        # Create BM25 index
        self.bm25 = BM25Okapi(self.tokenized_docs)
        print(f"✅ BM25 index created for {len(documents)} documents\n")
    
    def bm25_search(self, query, top_k=5):
        """Search using BM25 (keyword matching)"""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'index': idx,
                'score': scores[idx],
                'content': self.documents[idx]
            })
        
        return results
    
    def hybrid_search(self, query, vector_results, top_k=5):
        """
        Combine vector search + BM25 results
        Uses Reciprocal Rank Fusion (RRF)
        """
        print(f"🔀 Performing hybrid search for: '{query}'\n")
        
        # Get BM25 results
        bm25_results = self.bm25_search(query, top_k)
        
        # Normalize and combine scores (simple average)
        combined_scores = {}
        
        # Add vector scores
        for vec_result in vector_results:
            doc_id = vec_result.get('index', 0)
            combined_scores[doc_id] = {
                'vector_score': vec_result['score'],
                'bm25_score': 0,
                'content': vec_result['content']
            }
        
        # Add BM25 scores
        for bm25_result in bm25_results:
            doc_id = bm25_result['index']
            if doc_id not in combined_scores:
                combined_scores[doc_id] = {
                    'vector_score': 0,
                    'bm25_score': 0,
                    'content': bm25_result['content']
                }
            combined_scores[doc_id]['bm25_score'] = bm25_result['score']
        
        # Calculate combined score (weighted average)
        for doc_id in combined_scores:
            vector_score = combined_scores[doc_id]['vector_score']
            bm25_score = combined_scores[doc_id]['bm25_score'] / 100  # Normalize
            
            # Weighted combination
            combined_scores[doc_id]['combined_score'] = (
                0.6 * vector_score +  # 60% weight to vector
                0.4 * bm25_score      # 40% weight to BM25
            )
        
        # Sort by combined score
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1]['combined_score'],
            reverse=True
        )[:top_k]
        
        results = []
        for doc_id, scores_info in sorted_results:
            results.append({
                'index': doc_id,
                'content': scores_info['content'],
                'vector_score': scores_info['vector_score'],
                'bm25_score': scores_info['bm25_score'],
                'combined_score': scores_info['combined_score']
            })
        
        print(f"✅ Hybrid search returned {len(results)} results\n")
        return results
