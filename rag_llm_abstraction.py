"""
LLM abstraction layer supporting multiple backends:
- Google Vertex AI (Gemini)
- Azure OpenAI (GPT-4)
- Local models (Ollama, llama.cpp)
"""

import os
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM providers"""
    provider: str  # 'vertex', 'azure', 'local'
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 1024
    
    # Vertex AI specific
    project_id: Optional[str] = None
    location: Optional[str] = "us-central1"
    
    # Azure specific
    azure_endpoint: Optional[str] = None
    azure_api_key: Optional[str] = None
    
    # Local specific
    base_url: Optional[str] = "http://localhost:11434"  # Ollama default


@dataclass
class LLMResponse:
    """Standardized LLM response"""
    content: str
    model: str
    tokens_used: Optional[int] = None
    cost_estimate: Optional[float] = None
    latency_ms: Optional[float] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate response from prompt"""
        pass
    
    @abstractmethod
    def generate_with_context(
        self, 
        query: str, 
        context: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """Generate response with retrieved context"""
        pass


class VertexAILLM(LLMProvider):
    """Google Vertex AI LLM provider (Gemini)"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel
            
            vertexai.init(project=config.project_id, location=config.location)
            self.model = GenerativeModel(config.model_name)
            logger.info(f"Initialized Vertex AI LLM: {config.model_name}")
        except ImportError:
            raise ImportError("Install google-cloud-aiplatform")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate response"""
        import time
        start = time.time()
        
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        response = self.model.generate_content(
            full_prompt,
            generation_config={
                'temperature': self.config.temperature,
                'max_output_tokens': self.config.max_tokens
            }
        )
        
        latency_ms = (time.time() - start) * 1000
        
        # Estimate cost (Gemini Flash: ~$0.00025/1K chars input, ~$0.001/1K chars output)
        input_chars = len(full_prompt)
        output_chars = len(response.text)
        cost = (input_chars / 1000 * 0.00025) + (output_chars / 1000 * 0.001)
        
        return LLMResponse(
            content=response.text,
            model=self.config.model_name,
            tokens_used=None,  # Vertex doesn't always return this
            cost_estimate=cost,
            latency_ms=latency_ms
        )
    
    def generate_with_context(
        self, 
        query: str, 
        context: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """Generate with RAG context"""
        context_str = self._format_context(context)
        prompt = self._build_rag_prompt(query, context_str)
        return self.generate(prompt, system_prompt)
    
    @staticmethod
    def _format_context(context: List[Dict[str, str]]) -> str:
        """Format retrieved documents"""
        formatted = []
        for i, doc in enumerate(context, 1):
            formatted.append(f"[Document {i}]")
            formatted.append(f"Title: {doc.get('title', 'Untitled')}")
            formatted.append(f"Source: {doc.get('source', 'Unknown')}")
            formatted.append(f"Content: {doc['content']}")
            formatted.append("")
        return "\n".join(formatted)
    
    @staticmethod
    def _build_rag_prompt(query: str, context: str) -> str:
        """Build RAG prompt template - strict RAG for relevant context"""
        return f"""CONTEXT FROM KNOWLEDGE BASE:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
1. Answer using the information in the provided context
2. Cite sources using [Document X] notation when referencing specific information
3. If the context doesn't contain enough information, say "Based on the provided context, I cannot fully answer this question."
4. Be concise but comprehensive
5. If there's conflicting information in the context, acknowledge it

ANSWER:"""


class AzureOpenAILLM(LLMProvider):
    """Azure OpenAI LLM provider"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        try:
            from openai import AzureOpenAI
            
            self.client = AzureOpenAI(
                azure_endpoint=config.azure_endpoint,
                api_key=config.azure_api_key,
                api_version="2024-02-01"
            )
            logger.info(f"Initialized Azure OpenAI: {config.model_name}")
        except ImportError:
            raise ImportError("Install openai>=1.0.0")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate response"""
        import time
        start = time.time()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        latency_ms = (time.time() - start) * 1000
        
        # Cost estimation (GPT-4: ~$0.03/1K input tokens, ~$0.06/1K output tokens)
        tokens_used = response.usage.total_tokens
        cost = (response.usage.prompt_tokens / 1000 * 0.03) + \
               (response.usage.completion_tokens / 1000 * 0.06)
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=self.config.model_name,
            tokens_used=tokens_used,
            cost_estimate=cost,
            latency_ms=latency_ms
        )
    
    def generate_with_context(
        self, 
        query: str, 
        context: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """Generate with RAG context"""
        context_str = VertexAILLM._format_context(context)
        prompt = VertexAILLM._build_rag_prompt(query, context_str)
        return self.generate(prompt, system_prompt)


class LocalLLM(LLMProvider):
    """Local LLM provider using Ollama"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        try:
            import requests
            self.session = requests.Session()
            
            # Test connection
            response = self.session.get(f"{config.base_url}/api/tags")
            if response.status_code != 200:
                raise ConnectionError(f"Cannot connect to Ollama at {config.base_url}")
            
            logger.info(f"Initialized local LLM: {config.model_name}")
        except ImportError:
            raise ImportError("Install requests: pip install requests")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate response using Ollama"""
        import time
        import requests
        
        start = time.time()
        
        data = {
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens
            }
        }
        
        if system_prompt:
            data["system"] = system_prompt
        
        response = self.session.post(
            f"{self.config.base_url}/api/generate",
            json=data
        )
        response.raise_for_status()
        
        result = response.json()
        latency_ms = (time.time() - start) * 1000
        
        return LLMResponse(
            content=result['response'],
            model=self.config.model_name,
            tokens_used=result.get('eval_count', result.get('prompt_eval_count', 0)),
            cost_estimate=0.0,  # Local is free!
            latency_ms=latency_ms
        )
    
    def generate_with_context(
        self, 
        query: str, 
        context: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """Generate with RAG context"""
        context_str = VertexAILLM._format_context(context)
        prompt = VertexAILLM._build_rag_prompt(query, context_str)
        return self.generate(prompt, system_prompt)


class LLMFactory:
    """Factory for creating LLM providers"""
    
    @staticmethod
    def create(config: LLMConfig) -> LLMProvider:
        """Create LLM provider based on config"""
        providers = {
            'vertex': VertexAILLM,
            'azure': AzureOpenAILLM,
            'local': LocalLLM
        }
        
        provider_class = providers.get(config.provider)
        if not provider_class:
            raise ValueError(f"Unknown provider: {config.provider}")
        
        return provider_class(config)


# Example configurations
LLM_CONFIGS = {
    'vertex_gemini_flash': LLMConfig(
        provider='vertex',
        model_name='gemini-1.5-pro-002',  # Vertex AI Gemini 1.5 Pro
        temperature=0.7,
        max_tokens=1024,
        project_id=os.environ.get('GOOGLE_PROJECT_ID', 'your-project-id')
    ),
    'azure_gpt4': LLMConfig(
        provider='azure',
        model_name='gpt-4',
        temperature=0.7,
        max_tokens=1024,
        azure_endpoint='https://your-resource.openai.azure.com/',
        azure_api_key='your-api-key'
    ),
    'local_llama': LLMConfig(
        provider='local',
        model_name='llama3.2',  # 3B model, use llama3.2:1b for low memory
        temperature=0.7,
        max_tokens=1024,
        base_url=os.environ.get('OLLAMA_URL', 'http://localhost:11434')  # Use env var or localhost
    )
}


if __name__ == "__main__":
    # Test local LLM (requires Ollama running)
    config = LLM_CONFIGS['local_llama']
    llm = LLMFactory.create(config)
    
    response = llm.generate("What is machine learning?")
    print(f"Response: {response.content[:200]}...")
    print(f"Latency: {response.latency_ms:.0f}ms")
