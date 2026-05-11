"""Vector store infrastructure
"""

from app.repositories.vector_store import VectorStore


def create_vector_store(host: str, port: int, collection_name: str = "documents") -> VectorStore:
    """Create and return an initialized VectorStore instance."""
    return VectorStore(host=host, port=port, collection_name=collection_name)
