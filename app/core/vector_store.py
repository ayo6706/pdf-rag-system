from __future__ import annotations

from app.integrations.vectorstores import chroma


def create_vector_store(
    host: str,
    port: int,
    collection_name: str = "documents"
) -> chroma.VectorStore:
    """Create and return an initialized VectorStore instance.

    Args:
        host: Hostname of the ChromaDB server.
        port: Port of the ChromaDB server.
        collection_name: Name of the collection to use.

    Returns:
        An initialized VectorStore instance.
    """
    return chroma.VectorStore(host=host, port=port, collection_name=collection_name)
