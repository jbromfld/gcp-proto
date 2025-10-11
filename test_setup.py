"""Test script to verify RAG system setup"""

import sys
import requests
from elasticsearch import Elasticsearch

def test_elasticsearch():
    """Test Elasticsearch connection"""
    try:
        es = Elasticsearch(['http://localhost:9200'])
        health = es.cluster.health()
        print(f"✓ Elasticsearch: {health['status']}")
        return True
    except Exception as e:
        print(f"❌ Elasticsearch: {e}")
        return False

def test_ollama():
    """Test Ollama connection"""
    try:
        response = requests.get('http://localhost:11434/api/tags')
        models = response.json().get('models', [])
        if any('llama3.2' in m.get('name', '') for m in models):
            print("✓ Ollama: llama3.2 model available")
            return True
        else:
            print("❌ Ollama: llama3.2 model not found")
            return False
    except Exception as e:
        print(f"❌ Ollama: {e}")
        return False

def test_embeddings():
    """Test embedding generation"""
    try:
        from rag_embeddings import EmbeddingFactory, EMBEDDING_CONFIGS
        
        config = EMBEDDING_CONFIGS['local_minilm']
        embedder = EmbeddingFactory.create(config)
        
        embedding = embedder.embed_query("test query")
        print(f"✓ Embeddings: Generated {len(embedding)} dimensional vector")
        return True
    except Exception as e:
        print(f"❌ Embeddings: {e}")
        return False

def test_llm():
    """Test LLM generation"""
    try:
        from rag_llm_abstraction import LLMFactory, LLM_CONFIGS
        
        config = LLM_CONFIGS['local_llama']
        llm = LLMFactory.create(config)
        
        response = llm.generate("Say hello in one word")
        print(f"✓ LLM: Generated response ({len(response.content)} chars)")
        return True
    except Exception as e:
        print(f"❌ LLM: {e}")
        return False

def test_index():
    """Test if knowledge base index exists"""
    try:
        es = Elasticsearch(['http://localhost:9200'])
        
        if es.indices.exists(index='knowledge_base'):
            count = es.count(index='knowledge_base')['count']
            print(f"✓ Knowledge Base: {count} documents indexed")
            return True
        else:
            print("⚠️  Knowledge Base: Index not found (run ETL)")
            return True  # Not a failure, just needs data
    except Exception as e:
        print(f"❌ Knowledge Base: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing RAG System Setup...\n")
    
    tests = [
        ("Elasticsearch", test_elasticsearch),
        ("Ollama (Local LLM)", test_ollama),
        ("Embedding Model", test_embeddings),
        ("LLM Generation", test_llm),
        ("Knowledge Base Index", test_index)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\nTesting {name}:")
        results.append(test_func())
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ All tests passed! ({passed}/{total})")
        print("\n🎉 Your RAG system is ready to use!")
        print("   Run: streamlit run rag_ui.py")
        sys.exit(0)
    else:
        print(f"❌ Some tests failed ({passed}/{total} passed)")
        print("\n💡 Check the logs above and run setup.sh again")
        sys.exit(1)

if __name__ == "__main__":
    main()