import os
import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Dict, List, Union
from datetime import datetime
from mem0 import Memory

from OllamaMem.Memory import EnhancedMemory

DEFAULT_PERSIST_DIR='./mem_store/'


class OllamaMemoryBuilder:
    """
    Enhanced builder class for creating Ollama-based Memory instances with:
    - Automatic persistence detection and loading
    - User-based authentication and data isolation
    - Advanced querying with metadata filtering and custom prompts
    """

    def __init__(self):
        """Initialize builder with default values."""
        # Required parameters
        self._collection_name: Optional[str] = None
        self._persist_dir: Optional[str] = DEFAULT_PERSIST_DIR
        self._ollama_model: Optional[str] = None
        self._embedding_model: Optional[str] = None

        # Optional parameters with defaults
        self._max_tokens: int = 3000
        self._temperature: float = 0.0
        self._distance_metric: str = "cosine"
        self._ollama_base_url: str = "http://localhost:11434/"
        self.mem_version: float = 0.1

        # Optional: embedding dimensions (usually auto-detected)
        self._embedding_dims: Optional[int] = None

        # Application-layer features (not passed to mem0)
        self._retrieval_top_k: int = 5
        self._min_relevance_score: float = 0.7

        # Authentication
        self._enable_auth: bool = False
        self._auth_key: Optional[str] = None

        # Memory instance
        self._memory: Optional[Memory] = None
        self._current_user: Optional[str] = None

    # ==================== Builder Methods ====================

    def collection_name(self, name: str):
        """Set the collection name."""
        self._collection_name = name
        return self

    def persist_dir(self, path: str):
        """Set the persistence directory."""
        self._persist_dir = path
        return self

    def ollama_model(self, model: str):
        """Set the Ollama LLM model."""
        self._ollama_model = model
        return self

    def embedding_model(self, model: str):
        """Set the embedding model."""
        self._embedding_model = model
        return self

    def embedding_dims(self, dims: int):
        """Set the embedding dimensions (optional, usually auto-detected)."""
        self._embedding_dims = dims
        return self

    def max_tokens(self, tokens: int):
        """Set the maximum tokens for LLM generation."""
        self._max_tokens = tokens
        return self

    def temperature(self, temp: float):
        """Set the temperature for LLM generation."""
        self._temperature = temp
        return self

    def distance_metric(self, metric: str):
        """Set the distance metric for vector similarity."""
        self._distance_metric = metric
        return self

    def ollama_base_url(self, url: str):
        """Set the base URL for Ollama API."""
        self._ollama_base_url = url
        return self

    def retrieval_top_k(self, k: int):
        """Set the number of initial retrieval results."""
        self._retrieval_top_k = k
        return self

    def min_relevance_score(self, score: float):
        """Set the minimum relevance score threshold."""
        self._min_relevance_score = score
        return self

    def enable_authentication(self, auth_key: str):
        """
        Enable user-based authentication and data isolation.

        Args:
            auth_key: Secret key for generating user tokens
        """
        self._enable_auth = True
        self._auth_key = auth_key
        return self

    # ==================== Validation & Persistence ====================

    def _validate(self) -> None:
        """Validate that all required parameters are set."""
        required = {
            'collection_name': self._collection_name,
            'persist_dir': self._persist_dir,
            'ollama_model': self._ollama_model,
            'embedding_model': self._embedding_model
        }

        missing = [name for name, value in required.items() if value is None]

        if missing:
            raise ValueError(
                f"Missing required parameters: {', '.join(missing)}. "
                f"Please set them using the builder methods."
            )

    def _check_persistence_exists(self) -> bool:
        """
        Check if a valid FAISS index exists in the persist directory.

        Returns:
            True if valid index exists, False otherwise
        """
        if not self._persist_dir or not self._collection_name:
            return False

        persist_path = Path(self._persist_dir)
        if not persist_path.exists():
            return False

        # Check for FAISS index file
        index_file = persist_path / f"{self._collection_name}.index"
        metadata_file = persist_path / f"{self._collection_name}_metadata.json"

        return index_file.exists() and metadata_file.exists()

    def _create_persist_directory(self) -> None:
        """Create persistence directory if it doesn't exist."""
        if self._persist_dir:
            Path(self._persist_dir).mkdir(parents=True, exist_ok=True)

    def _get_persistence_info(self) -> Dict[str, Any]:
        """Get information about the persisted index."""
        if not self._check_persistence_exists():
            return {"exists": False}

        persist_path = Path(self._persist_dir)
        index_file = persist_path / f"{self._collection_name}.index"

        return {
            "exists": True,
            "size_mb": index_file.stat().st_size / (1024 * 1024),
            "path": str(index_file),
            "modified": datetime.fromtimestamp(index_file.stat().st_mtime).isoformat()
        }

    # ==================== Authentication ====================

    def _generate_user_token(self, user_id: str) -> str:
        """
        Generate a secure token for user authentication.

        Args:
            user_id: User identifier

        Returns:
            Hashed token for the user
        """
        if not self._auth_key:
            raise ValueError("Authentication key not set. Call enable_authentication() first.")

        combined = f"{user_id}:{self._auth_key}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def _validate_user_token(self, user_id: str, token: str) -> bool:
        """
        Validate a user token.

        Args:
            user_id: User identifier
            token: Token to validate

        Returns:
            True if token is valid, False otherwise
        """
        expected_token = self._generate_user_token(user_id)
        return token == expected_token

    def authenticate_user(self, user_id: str) -> str:
        """
        Authenticate a user and return their access token.

        Args:
            user_id: User identifier (e.g., "alice", "bob")

        Returns:
            Access token for the user

        Raises:
            ValueError: If authentication is not enabled
        """
        if not self._enable_auth:
            raise ValueError("Authentication not enabled. Call enable_authentication() first.")

        self._current_user = user_id
        return self._generate_user_token(user_id)

    def _get_user_metadata_filter(self, user_id: str) -> Dict[str, str]:
        """
        Get metadata filter for user isolation.

        Args:
            user_id: User identifier

        Returns:
            Metadata filter dictionary
        """
        return {"user_id": user_id}

    # ==================== Build Methods ====================

    def build_config(self) -> Dict[str, Any]:
        """
        Build and return the memory configuration dictionary.

        Returns:
            Configuration dictionary for Memory.from_config()

        Raises:
            ValueError: If required parameters are not set
        """
        self._validate()

        # Build FAISS config with only supported fields
        faiss_config = {
            "collection_name": self._collection_name,
            "path": self._persist_dir,
            "distance_strategy": self._distance_metric
        }

        # Add optional embedding dimensions if specified
        if self._embedding_dims:
            faiss_config["embedding_model_dims"] = self._embedding_dims

        mem_conf = {
            "vector_store": {
                "provider": "faiss",
                "config": faiss_config
            },
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": self._ollama_model,
                    "temperature": self._temperature,
                    "max_tokens": self._max_tokens,
                    "ollama_base_url": self._ollama_base_url
                }
            },
            "embedder": {
                "provider": "ollama",
                "config": {
                    "model": self._embedding_model,
                    "ollama_base_url": self._ollama_base_url
                }
            }
        }

        # Add version metadata
        mem_conf["version"] = f"v{self.mem_version}"

        # Note: Reranking, advanced HNSW params, and cleanup policies are handled
        # at the application layer in EnhancedMemory, not in mem0 config

        return mem_conf

    def build(self, verbose: bool = True) -> 'EnhancedMemory':
        """
        Build and return an EnhancedMemory instance.
        Automatically detects existing persistence and loads or creates new instance.

        Args:
            verbose: Print status messages

        Returns:
            Configured EnhancedMemory instance

        Raises:
            ValueError: If required parameters are not set
        """
        self._validate()
        self._create_persist_directory()

        persistence_exists = self._check_persistence_exists()

        if verbose:
            if persistence_exists:
                info = self._get_persistence_info()
                print(f"✓ Loading existing FAISS index from {self._persist_dir}")
                print(f"  Size: {info['size_mb']:.2f} MB | Last modified: {info['modified']}")
            else:
                print(f"✓ Creating new FAISS index at {self._persist_dir}")

        config = self.build_config()
        memory = Memory.from_config(config)

        # Wrap in EnhancedMemory for additional features
        enhanced_memory = EnhancedMemory(
            memory=memory,
            enable_auth=self._enable_auth,
            auth_key=self._auth_key,
            persist_dir=self._persist_dir,
            collection_name=self._collection_name
        )

        return enhanced_memory


