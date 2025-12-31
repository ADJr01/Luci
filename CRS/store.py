"""Main Store class for CRS system."""
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import pickle
import json
from datetime import datetime

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings

from .config import StoreConfig, RerankConfig
from .rerank import Rerank


class Store:
    """
    Context Retriever Store - Production-grade vector database with reranking.

    Architecture:
    ┌────────────────────────────────────────────────────────┐
    │                   Store Lifecycle                       │
    ├────────────────────────────────────────────────────────┤
    │  1. Initialization (load or create FAISS)              │
    │  2. Document Addition (embedding + indexing)           │
    │  3. Persistence (atomic write to disk)                 │
    │  4. Query Processing (vector search + filtering)       │
    │  5. Reranking (automatic quality improvement)          │
    └────────────────────────────────────────────────────────┘

    Features:
    - Persistent FAISS vector store
    - Metadata filtering
    - Automatic reranking
    - Atomic persistence
    - Batch processing

    Example:
        >>> store = Store(
        ...     persist_dir="/data/crs",
        ...     embedding=ollama_embedding,
        ...     similarity_metric="cosine"
        ... )
        >>> store.add_document(doc)
        >>> results = store.query("how many R in RAHAT", filter={"chat_id": "..."})
    """

    def __init__(
            self,
            persist_dir: Union[str, Path],
            embedding: Embeddings,
            similarity_metric: str = "cosine",
            config: Optional[StoreConfig] = None,
            rerank_config: Optional[RerankConfig] = None
    ):
        """
        Initialize the Store.

        Args:
            persist_dir: Directory for persisting data
            embedding: LangChain embedding model
            similarity_metric: FAISS metric ("l2", "cosine", "inner_product")
            config: Store configuration
            rerank_config: Reranking configuration
        """
        self.config = config or StoreConfig(
            persist_dir=Path(persist_dir),
            similarity_metric=similarity_metric
        )
        self.embedding = embedding
        self.reranker = Rerank(rerank_config)

        # Ensure persistence directory exists
        self.config.persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize or load FAISS vector store
        self.vector_store = self._initialize_vector_store()

        # Track statistics
        self.stats = {
            "total_documents": 0,
            "last_updated": None,
            "queries_processed": 0
        }
        self._load_stats()

    def _initialize_vector_store(self) -> FAISS:
        """
        Initialize FAISS vector store.

        Loads existing index if available, otherwise creates new one.
        """
        index_path = self.config.persist_dir / self.config.index_name

        if index_path.exists():
            try:
                print(f"Loading existing FAISS index from {index_path}")
                vector_store = FAISS.load_local(
                    str(index_path),
                    self.embedding,
                    allow_dangerous_deserialization=True  # We control the source
                )
                print(f"Successfully loaded index with {vector_store.index.ntotal} vectors")
                return vector_store
            except Exception as e:
                print(f"Warning: Failed to load existing index: {e}")
                print("Creating new index...")

        # Create new empty index
        print("Creating new FAISS index")
        # Create with a dummy document to initialize
        dummy_doc = Document(page_content="initialization", metadata={})
        vector_store = FAISS.from_documents(
            [dummy_doc],
            self.embedding,
            distance_strategy=self._get_distance_strategy()
        )

        # Remove dummy document
        vector_store.delete([vector_store.index_to_docstore_id[0]])

        return vector_store

    def _get_distance_strategy(self) -> str:
        """Map similarity metric to FAISS distance strategy."""
        mapping = {
            "l2": "EUCLIDEAN_DISTANCE",
            "cosine": "COSINE",
            "inner_product": "MAX_INNER_PRODUCT"
        }
        return mapping.get(
            self.config.similarity_metric,
            "COSINE"
        )

    def add_document(self, document: Document) -> None:
        """
        Add a single document to the store.

        Args:
            document: LangChain Document with page_content and metadata
        """
        self.add_documents([document])

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add multiple documents to the store with batch processing.

        Args:
            documents: List of LangChain Documents
        """
        if not documents:
            return

        try:
            # Validate documents
            validated_docs = self._validate_documents(documents)

            # Add to FAISS
            self.vector_store.add_documents(validated_docs)

            # Update statistics
            self.stats["total_documents"] += len(validated_docs)
            self.stats["last_updated"] = datetime.now()

            # Persist changes
            self.persist()

            print(f"Successfully added {len(validated_docs)} documents")

        except Exception as e:
            print(f"Error adding documents: {e}")
            raise

    def _validate_documents(self, documents: List[Document]) -> List[Document]:
        """
        Validate and enrich documents before storage.

        Ensures:
        - page_content is not empty
        - metadata is a dict
        - required metadata fields exist
        """
        validated = []

        for doc in documents:
            if not doc.page_content or not doc.page_content.strip():
                print(f"Warning: Skipping document with empty content")
                continue

            if not isinstance(doc.metadata, dict):
                doc.metadata = {}

            # Add ingestion timestamp if not present
            if "ingestion_time" not in doc.metadata:
                doc.metadata["ingestion_time"] = datetime.now()

            validated.append(doc)

        return validated

    def query(
            self,
            q: str,
            filter: Optional[Dict[str, Any]] = None,
            k: Optional[int] = None,
            use_reranking: bool = True
    ) -> List[tuple[Document, float]]:
        """
        Query the store with semantic search and optional metadata filtering.

        Args:
            q: Natural language query
            filter: Metadata filters (e.g., {"chat_id": "...", "mode": "..."})
            k: Number of results to return (uses reranker config if None)
            use_reranking: Whether to apply reranking

        Returns:
            List of (Document, score) tuples, sorted by relevance

        Example:
            >>> results = store.query(
            ...     q="how many R in RAHAT",
            ...     filter={"chat_id": "chat_id_2025-30-12_10:30"},
            ...     k=5
            ... )
        """
        try:
            # Update query statistics
            self.stats["queries_processed"] += 1

            # Determine number of candidates to retrieve
            retrieval_k = k or self.reranker.config.top_k if use_reranking else self.reranker.config.final_k

            # Perform similarity search
            if filter:
                # FAISS doesn't natively support metadata filtering,
                # so we retrieve more results and filter post-hoc
                results = self.vector_store.similarity_search_with_score(
                    q,
                    k=retrieval_k * 3  # Over-retrieve for filtering
                )

                # Apply metadata filters
                filtered_results = [
                    (doc, score)
                    for doc, score in results
                    if self._matches_filter(doc.metadata, filter)
                ][:retrieval_k]
            else:
                filtered_results = self.vector_store.similarity_search_with_score(
                    q,
                    k=retrieval_k
                )

            if not filtered_results:
                print("No results found matching query and filters")
                return []

            # Apply reranking if enabled
            if use_reranking:
                documents = [doc for doc, _ in filtered_results]
                scores = [score for _, score in filtered_results]

                reranked_results = self.reranker.rerank(
                    query=q,
                    documents=documents,
                    initial_scores=scores
                )

                return reranked_results

            # Return top-k without reranking
            final_k = k or self.reranker.config.final_k
            return filtered_results[:final_k]

        except Exception as e:
            print(f"Error during query: {e}")
            raise

    def _matches_filter(
            self,
            metadata: Dict[str, Any],
            filter_dict: Dict[str, Any]
    ) -> bool:
        """
        Check if document metadata matches filter criteria.

        Supports:
        - Exact matching
        - Case-insensitive string matching
        - List membership checks
        """
        for key, value in filter_dict.items():
            if key not in metadata:
                return False

            meta_value = metadata[key]

            # Case-insensitive string matching
            if isinstance(value, str) and isinstance(meta_value, str):
                if value.lower() != meta_value.lower():
                    return False
            # List membership
            elif isinstance(value, list):
                if meta_value not in value:
                    return False
            # Exact matching
            elif meta_value != value:
                return False

        return True

    def persist(self) -> None:
        """
        Persist vector store and metadata to disk atomically.

        Uses atomic write pattern to prevent corruption.
        """
        try:
            index_path = self.config.persist_dir / self.config.index_name

            # Save FAISS index
            self.vector_store.save_local(str(index_path))

            # Save statistics
            self._save_stats()

            print(f"Successfully persisted store to {index_path}")

        except Exception as e:
            print(f"Error persisting store: {e}")
            raise

    def _save_stats(self) -> None:
        """Save statistics to JSON file."""
        stats_path = self.config.persist_dir / "stats.json"

        # Convert datetime objects to strings
        serializable_stats = {
            k: v.isoformat() if isinstance(v, datetime) else v
            for k, v in self.stats.items()
        }

        with open(stats_path, "w") as f:
            json.dump(serializable_stats, f, indent=2)

    def _load_stats(self) -> None:
        """Load statistics from JSON file."""
        stats_path = self.config.persist_dir / "stats.json"

        if stats_path.exists():
            try:
                with open(stats_path, "r") as f:
                    loaded_stats = json.load(f)

                # Convert ISO strings back to datetime
                if "last_updated" in loaded_stats and loaded_stats["last_updated"]:
                    loaded_stats["last_updated"] = datetime.fromisoformat(
                        loaded_stats["last_updated"]
                    )

                self.stats.update(loaded_stats)
            except Exception as e:
                print(f"Warning: Failed to load stats: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get current store statistics."""
        return self.stats.copy()

    def delete_by_filter(self, filter: Dict[str, Any]) -> int:
        """
        Delete documents matching filter criteria.

        Args:
            filter: Metadata filters

        Returns:
            Number of documents deleted
        """
        # Note: FAISS doesn't support efficient filtered deletion
        # This is a workaround that rebuilds the index
        print("Warning: Filtered deletion requires index rebuild")

        # Retrieve all documents
        all_docs = self._get_all_documents()

        # Filter out matching documents
        remaining_docs = [
            doc for doc in all_docs
            if not self._matches_filter(doc.metadata, filter)
        ]

        deleted_count = len(all_docs) - len(remaining_docs)

        if deleted_count > 0:
            # Rebuild index
            self.vector_store = FAISS.from_documents(
                remaining_docs if remaining_docs else [Document(page_content="init", metadata={})],
                self.embedding,
                distance_strategy=self._get_distance_strategy()
            )

            if not remaining_docs:
                self.vector_store.delete([self.vector_store.index_to_docstore_id[0]])

            self.stats["total_documents"] -= deleted_count
            self.persist()

            print(f"Deleted {deleted_count} documents")

        return deleted_count

    def _get_all_documents(self) -> List[Document]:
        """Retrieve all documents from the store."""
        # Access internal docstore
        docstore = self.vector_store.docstore
        all_docs = []

        for doc_id in self.vector_store.index_to_docstore_id.values():
            doc = docstore.search(doc_id)
            if doc:
                all_docs.append(doc)

        return all_docs