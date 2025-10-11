"""
Embedding abstraction layer supporting multiple backends:
- Google Vertex AI (production)
- Azure OpenAI (alternative cloud)
- Local models (testing/development)
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models"""
    provider: str  # 'vertex', 'azure', 'local'
    model_name: str
    dimensions: int
    max_batch_size: int = 32
    
    # Vertex AI specific
    project_id: Optional[str] = None
    location: Optional[str] = "us-central1"
    
    # Azure specific
    azure_endpoint: Optional[str] = None
    azure_api_key: Optional[str] = None
    
    # Local model specific
    local_model_path: Optional[str] = None


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts"""
        pass
    
    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query (may have different processing)"""
        pass
    
    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return embedding dimensions"""
        pass


class VertexAIEmbeddings(EmbeddingProvider):
    """Google Vertex AI embedding provider"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        try:
            from vertexai.language_models import TextEmbeddingModel
            import vertexai
            
            vertexai.init(project=config.project_id, location=config.location)
            self.model = TextEmbeddingModel.from_pretrained(config.model_name)
            logger.info(f"Initialized Vertex AI embeddings: {config.model_name}")
        except ImportError:
            raise ImportError("Install google-cloud-aiplatform: pip install google-cloud-aiplatform")
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed texts in batches"""
        embeddings = []
        batch_size = self.config.max_batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.get_embeddings(batch)
            embeddings.extend([emb.values for emb in batch_embeddings])
        
        return np.array(embeddings)
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query"""
        embedding = self.model.get_embeddings([query])[0]
        return np.array(embedding.values)
    
    @property
    def dimensions(self) -> int:
        return self.config.dimensions


class AzureOpenAIEmbeddings(EmbeddingProvider):
    """Azure OpenAI embedding provider"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        try:
            from openai import AzureOpenAI
            
            self.client = AzureOpenAI(
                azure_endpoint=config.azure_endpoint,
                api_key=config.azure_api_key,
                api_version="2024-02-01"
            )
            logger.info(f"Initialized Azure OpenAI embeddings: {config.model_name}")
        except ImportError:
            raise ImportError("Install openai: pip install openai>=1.0.0")
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed texts in batches"""
        embeddings = []
        batch_size = self.config.max_batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                model=self.config.model_name,
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query"""
        response = self.client.embeddings.create(
            model=self.config.model_name,
            input=[query]
        )
        return np.array(response.data[0].embedding)
    
    @property
    def dimensions(self) -> int:
        return self.config.dimensions


class LocalEmbeddings(EmbeddingProvider):
    """Local embedding provider using sentence-transformers"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        try:
            from sentence_transformers import SentenceTransformer
            
            model_name = config.local_model_path or config.model_name
            self.model = SentenceTransformer(model_name)
            self._dimensions = self.model.get_sentence_embedding_dimension()
            logger.info(f"Initialized local embeddings: {model_name}")
        except ImportError:
            raise ImportError("Install sentence-transformers: pip install sentence-transformers")
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed texts"""
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.max_batch_size,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query"""
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )[0]
        return embedding
    
    @property
    def dimensions(self) -> int:
        return self._dimensions


class EmbeddingFactory:
    """Factory for creating embedding providers"""
    
    @staticmethod
    def create(config: EmbeddingConfig) -> EmbeddingProvider:
        """Create embedding provider based on config"""
        providers = {
            'vertex': VertexAIEmbeddings,
            'azure': AzureOpenAIEmbeddings,
            'local': LocalEmbeddings
        }
        
        provider_class = providers.get(config.provider)
        if not provider_class:
            raise ValueError(f"Unknown provider: {config.provider}")
        
        return provider_class(config)


# Example configurations
EMBEDDING_CONFIGS = {
    'vertex_gecko': EmbeddingConfig(
        provider='vertex',
        model_name='textembedding-gecko@003',
        dimensions=768,
        project_id='your-project-id',
        location='us-central1'
    ),
    'azure_ada': EmbeddingConfig(
        provider='azure',
        model_name='text-embedding-ada-002',
        dimensions=1536,
        azure_endpoint='https://your-resource.openai.azure.com/',
        azure_api_key='your-api-key'
    ),
    'local_minilm': EmbeddingConfig(
        provider='local',
        model_name='all-MiniLM-L6-v2',  # Fast, 384 dims
        dimensions=384
    ),
    'local_mpnet': EmbeddingConfig(
        provider='local',
        model_name='all-mpnet-base-v2',  # Better quality, 768 dims
        dimensions=768
    )
}


if __name__ == "__main__":
    # Test local embeddings
    config = EMBEDDING_CONFIGS['local_minilm']
    embedder = EmbeddingFactory.create(config)
    
    texts = ["Hello world", "Machine learning is great"]
    embeddings = embedder.embed_texts(texts)
    print(f"Embedded {len(texts)} texts: {embeddings.shape}")
    
    query = "What is AI?"
    query_emb = embedder.embed_query(query)
    print(f"Query embedding shape: {query_emb.shape}")
