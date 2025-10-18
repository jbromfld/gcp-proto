"""
Elasticsearch client factory with authentication support
"""
import os
from elasticsearch import Elasticsearch
from typing import Optional


def create_elasticsearch_client(
    url: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    cloud_id: Optional[str] = None
) -> Elasticsearch:
    """
    Create Elasticsearch client with proper authentication
    
    Supports:
    - Local/unsecured: http://localhost:9200
    - Elastic Cloud: cloud_id + password
    - Self-hosted with auth: https://url + username + password
    
    Args:
        url: Elasticsearch URL (http:// or https://)
        username: Username for authentication
        password: Password for authentication
        cloud_id: Elastic Cloud ID (alternative to URL)
    
    Returns:
        Configured Elasticsearch client
    """
    # Get from environment if not provided
    url = url or os.environ.get('ELASTICSEARCH_URL', 'http://elasticsearch:9200')
    username = username or os.environ.get('ELASTICSEARCH_USERNAME', 'elastic')
    password = password or os.environ.get('ELASTICSEARCH_PASSWORD')
    cloud_id = cloud_id or os.environ.get('ELASTICSEARCH_CLOUD_ID')
    
    # Type narrowing
    assert url is not None, "Elasticsearch URL must be provided"
    
    # Elastic Cloud (using cloud_id)
    if cloud_id:
        if not password or not username:
            raise ValueError("Username and password required for Elastic Cloud")
        
        return Elasticsearch(
            cloud_id=cloud_id,
            basic_auth=(username, password),
            request_timeout=30,
            max_retries=3,
            retry_on_timeout=True
        )
    
    # Self-hosted with HTTPS (requires auth)
    elif url.startswith('https://'):
        if not password or not username:
            raise ValueError("Username and password required for HTTPS Elasticsearch")
        
        return Elasticsearch(
            [url],
            basic_auth=(username, password),
            request_timeout=30,
            max_retries=3,
            retry_on_timeout=True,
            verify_certs=True
        )
    
    # Local development (no auth)
    else:
        return Elasticsearch(
            [url],
            request_timeout=30,
            max_retries=3,
            retry_on_timeout=True
        )


if __name__ == "__main__":
    # Test connection
    import sys
    
    try:
        es = create_elasticsearch_client()
        health = es.cluster.health()
        print("✓ Connected to Elasticsearch")
        print(f"  Cluster: {health['cluster_name']}")
        print(f"  Status: {health['status']}")
        print(f"  Nodes: {health['number_of_nodes']}")
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        sys.exit(1)

