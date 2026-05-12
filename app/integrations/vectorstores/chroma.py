"""ChromaDB vector store wrapper for chunk storage and retrieval.

Provides upsert and deletion operations for document chunks, connecting
to a standalone ChromaDB server via HTTP.
"""

import uuid
import logging
import chromadb

from dataclasses import dataclass

from app.schemas.embedding import ChunkWithEmbedding

@dataclass
class SearchResult:
    chunk_text: str
    doc_id: str
    page_number: int
    distance: float
    similarity: float

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
            "VectorStore initialized: collection='%s', server='http://%s:%s'",
            collection_name, host, port
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
        logger.info("Upserted %d chunks for doc_id=%s", len(chunks), doc_id)

    def delete_by_doc_id(self, doc_id: str) -> None:
        """Delete all chunks belonging to a document.

        Args:
            doc_id: The document's UUID string.
        """
        try:
            self.collection.delete(where={"doc_id": doc_id})
            logger.info("Deleted chunks for doc_id=%s", doc_id)
        except Exception as exc:
            exc_msg = str(exc).lower()
            if "no matching" in exc_msg or "not found" in exc_msg or "empty" in exc_msg:
                logger.debug("No existing chunks to delete for doc_id=%s", doc_id)
            else:
                logger.exception("Unexpected error deleting chunks for doc_id=%s", doc_id)
                raise

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        doc_ids: list[str] | None = None,
    ) -> list[SearchResult]:
        """Search for similar chunks. Optionally filter by doc_ids.

        Args:
            query_embedding: The embedding of the search query.
            top_k: Number of results to return.
            doc_ids: Optional list of document UUID strings to filter by.

        Returns:
            A list of SearchResult objects, sorted by similarity.
        """
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
        }
        
        if doc_ids:
            if len(doc_ids) == 1:
                kwargs["where"] = {"doc_id": doc_ids[0]}
            else:
                kwargs["where"] = {"doc_id": {"$in": doc_ids}}
                
        results = self.collection.query(**kwargs)
        
        search_results = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return search_results
            
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            similarity = 1 - (distance / 2.0)
            meta = results["metadatas"][0][i]
            
            search_results.append(
                SearchResult(
                    chunk_text=results["documents"][0][i],
                    doc_id=meta.get("doc_id", ""),
                    page_number=meta.get("page_number", 0),
                    distance=distance,
                    similarity=max(0.0, min(1.0, similarity)), # clamp between 0 and 1
                )
            )
            
        return search_results
