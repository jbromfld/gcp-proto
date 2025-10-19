#!/usr/bin/env python3
"""
Debug script to understand why similarity search is vague
"""

import requests
import json

def debug_search_query(query: str, api_url: str = "https://rag-api-XXXXX-uc.a.run.app"):
    """Debug a search query to see what's happening"""
    
    print(f"🔍 Debugging query: '{query}'")
    print("=" * 60)
    
    # Make the API call
    try:
        response = requests.post(
            f"{api_url}/api/query",
            json={"query": query, "top_k": 5},
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"❌ API Error: {response.status_code}")
            print(response.text)
            return
        
        data = response.json()
        
        # Show retrieved documents
        print(f"📊 Retrieved {len(data.get('retrieved_docs', []))} documents")
        print(f"⏱️  Retrieval time: {data.get('retrieval_time_ms', 0):.0f}ms")
        print(f"⏱️  Generation time: {data.get('generation_time_ms', 0):.0f}ms")
        print()
        
        # Show each retrieved document
        for i, doc in enumerate(data.get('retrieved_docs', []), 1):
            print(f"📄 Document {i}:")
            print(f"   Title: {doc.get('title', 'N/A')}")
            print(f"   Score: {doc.get('score', 0):.2f}")
            print(f"   Source: {doc.get('source_url', 'N/A')}")
            print(f"   Content: {doc.get('content', '')[:200]}...")
            print()
        
        # Show the answer
        print(f"💬 Answer:")
        print(data.get('answer', 'No answer'))
        print()
        
        # Check if using general knowledge fallback
        if len(data.get('retrieved_docs', [])) == 0:
            print("⚠️  WARNING: No documents retrieved - using general knowledge fallback")
        else:
            max_score = max(doc.get('score', 0) for doc in data.get('retrieved_docs', []))
            print(f"📈 Max relevance score: {max_score:.2f}")
            if max_score < 3.0:
                print("⚠️  WARNING: Low relevance score - might be using general knowledge fallback")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Test the FastAPI features query
    debug_search_query("What are the features of FastAPI")
