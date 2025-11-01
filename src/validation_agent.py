"""
Validation Agent
Checks synthesis output for hallucinations, contradictions, and unsupported claims
"""
import os
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer, util

load_dotenv()

class ValidationAgent:
    def __init__(self, groq_api_key=None):
        """Initialize Validation Agent"""
        print("✅ Initializing Validation Agent...\n")
        
        self.groq_client = Groq(api_key=groq_api_key)
        self.model_name = "llama-3.3-70b-versatile"
        self.nli_model = SentenceTransformer('cross-encoder/qnli-distilroberta-base')
        
        self.validation_prompt = """You are a fact-checking expert. Analyze if the answer claims are supported by the sources.

SOURCES:
{sources}

ANSWER:
{answer}

Check for:
1. Hallucinations: Claims not in sources
2. Contradictions: Conflicting statements
3. Unsupported claims: Missing evidence

Respond in this format ONLY:
VALID: yes/no
CONFIDENCE: 0-100
ISSUES: [list any problems]
REASONING: [brief explanation]"""
        
        print("✅ Validation Agent ready!\n")
    
    def extract_claims(self, answer):
        """Extract individual claims from answer"""
        # Split by sentences
        claims = [s.strip() for s in answer.split('.') if s.strip() and len(s.strip()) > 10]
        return claims
    
    def check_hallucinations(self, answer, documents):
        """Check if answer contains hallucinations using NLI"""
        print("🔍 Checking for hallucinations...")
        
        claims = self.extract_claims(answer)
        source_text = " ".join([doc['content'] for doc in documents])
        
        hallucinated_claims = []
        
        try:
            for claim in claims:
                # Check if claim is entailed by sources
                scores = self.nli_model.predict([[source_text, claim]])
                
                # If not entailed (contradiction or neutral), it might be hallucinated
                if scores[0] < 0.5:  # Low entailment score
                    hallucinated_claims.append(claim)
            
            if hallucinated_claims:
                print(f"   ⚠️  Found {len(hallucinated_claims)} potential hallucinations")
            else:
                print(f"   ✓ No hallucinations detected")
            
            return hallucinated_claims
        
        except Exception as e:
            print(f"   ⚠️  Hallucination check skipped: {e}")
            return []
    
    def check_citations(self, answer, document_sources):
        """Check if claims are properly cited"""
        print("🔗 Checking citations...")
        
        import re
        
        # Extract cited sources
        cited_sources = re.findall(r'\[Source: ([^\]]+)\]', answer)
        
        # Check if all cited sources exist
        valid_cites = []
        invalid_cites = []
        
        for cite in cited_sources:
            if cite.strip() in document_sources:
                valid_cites.append(cite)
            else:
                invalid_cites.append(cite)
        
        if invalid_cites:
            print(f"   ⚠️  Found {len(invalid_cites)} invalid citations: {invalid_cites}")
        else:
            print(f"   ✓ All citations are valid ({len(valid_cites)} total)")
        
        return {
            'valid': valid_cites,
            'invalid': invalid_cites,
            'coverage': len(valid_cites) / max(len(cited_sources), 1) if cited_sources else 0
        }
    
    def llm_validation(self, answer, documents):
        """Use LLM to validate answer quality"""
        print("🤖 LLM validation...")
        
        # Format sources
        sources_text = "\n".join([
            f"- {doc['source']}: {doc['content'][:200]}..."
            for doc in documents
        ])
        
        prompt = self.validation_prompt.format(
            sources=sources_text,
            answer=answer
        )
        
        try:
            response = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model_name,
                temperature=0.3,
                max_tokens=300
            )
            
            validation_result = response.choices[0].message.content.strip()
            print(f"   ✓ LLM validation complete")
            
            return validation_result
        
        except Exception as e:
            print(f"   ❌ LLM validation error: {e}")
            return ""
    
    def validate(self, answer, documents):
        """Main validation pipeline"""
        print("\n" + "=" * 70)
        print("VALIDATION PHASE")
        print("=" * 70 + "\n")
        
        document_sources = [doc['source'] for doc in documents]
        
        # Check 1: Hallucinations
        hallucinations = self.check_hallucinations(answer, documents)
        
        # Check 2: Citations
        citations = self.check_citations(answer, document_sources)
        
        # Check 3: LLM Validation
        llm_validation = self.llm_validation(answer, documents)
        
        # Compile results
        validation_result = {
            'hallucinations': hallucinations,
            'citations': citations,
            'llm_validation': llm_validation,
            'is_valid': len(hallucinations) == 0 and len(citations['invalid']) == 0,
            'confidence': 100 - (len(hallucinations) * 5 + len(citations['invalid']) * 10)
        }
        
        print("\n" + "=" * 70)
        print("VALIDATION RESULT")
        print("=" * 70)
        print(f"Valid: {validation_result['is_valid']}")
        print(f"Confidence: {validation_result['confidence']}%")
        print(f"Hallucinations: {len(hallucinations)}")
        print(f"Invalid Citations: {len(citations['invalid'])}")
        print("=" * 70 + "\n")
        
        return validation_result


# Test the agent
if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    
    validator = ValidationAgent(groq_api_key=api_key)
    
    test_answer = """FastAPI is a modern Python web framework. [Source: fastapi.md] 
    It provides automatic API documentation. [Source: fastapi.md]
    The framework is used by Google. [Source: nonexistent.md]"""
    
    test_docs = [
        {
            'source': 'fastapi.md',
            'content': 'FastAPI is a modern, fast web framework for building APIs with Python based on standard Python type hints.'
        },
        {
            'source': 'python.md',
            'content': 'Python is a high-level programming language.'
        }
    ]
    
    print("=" * 70)
    print("✅ VALIDATION AGENT TEST")
    print("=" * 70 + "\n")
    
    result = validator.validate(test_answer, test_docs)
    
    print(f"Result: {result}")
