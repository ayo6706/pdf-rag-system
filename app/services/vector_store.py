"""ChromaDB vector store wrapper for chunk storage and retrieval.

Provides upsert and deletion operations for document chunks, connecting
to a standalone ChromaDB server via HTTP.
"""

import uuid
import logging
import chromadb

from app.services.llm import ChunkWithEmbedding

logger = logging.getLogger(__name__)

# Stable namespace for deterministic chunk IDs
_CHUNK_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


class VectorStore:
    """Wrapper around a ChromaDB HttpClient."""

    def __init__(self, host: str = "localhost", port: int = 8000, collection_name: str = "documents"):
        """Initialize the vector store.

        Args:
            host: Hostname of the ChromaDB server.
            port: Port of the ChromaDB server.
            collection_name: Name of the ChromaDB collection.
        """
        # We use HttpClient to connect to a standalone server (e.g. via Docker)
        self.client = chromadb.HttpClient(host=host, port=port)
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"VectorStore initialized: collection='{collection_name}', "
            f"server='http://{host}:{port}'"
        )

    def upsert_chunks(self, doc_id: str, chunks: list[ChunkWithEmbedding]) -> None:
        """Upsert chunk embeddings with metadata into ChromaDB.

        Each chunk gets a deterministic UUID derived from its doc_id and
        chunk_index, ensuring idempotent upserts without needing to
        delete first.

        Args:
            doc_id: The document's UUID string.
            chunks: List of chunks with their embeddings.
        """
        if not chunks:
            return

        ids = [
            str(uuid.uuid5(_CHUNK_NS, f"{doc_id}:{c.chunk.chunk_index}"))
            for c in chunks
        ]
        embeddings = [c.embedding for c in chunks]
        documents = [c.chunk.text for c in chunks]
        metadatas = [
            {
                "doc_id": doc_id,
                "page_number": c.chunk.page_number,
                "chunk_index": c.chunk.chunk_index,
            }
            for c in chunks
        ]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info(f"Upserted {len(chunks)} chunks for doc_id={doc_id}")

    def delete_by_doc_id(self, doc_id: str) -> None:
        """Delete all chunks belonging to a document.

        Args:
            doc_id: The document's UUID string.
        """
        try:
            self.collection.delete(where={"doc_id": doc_id})
            logger.info(f"Deleted chunks for doc_id={doc_id}")
        except Exception as exc:
            exc_msg = str(exc).lower()
            if "no matching" in exc_msg or "not found" in exc_msg or "empty" in exc_msg:
                logger.debug(f"No existing chunks to delete for doc_id={doc_id}")
            else:
                logger.exception(f"Unexpected error deleting chunks for doc_id={doc_id}")
                raise
