import os
import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Dict, List, Union
from datetime import datetime
from mem0 import Memory


class SessionNotSetError(Exception):
    """Exception raised when attempting operations without an active user session."""
    pass


class EnhancedMemory:
    """
    Enhanced Memory wrapper with session management and advanced querying capabilities.
    """

    def __init__(
            self,
            memory: Memory,
            persist_dir: Optional[str] = None,
            collection_name: Optional[str] = None,
            re_ranking_model: Optional[str] = None,
            re_rank_top_k: int = 15,
            enable_reranking: bool = False
    ):
        """
        Initialize EnhancedMemory.

        Args:
            memory: Base Memory instance
            persist_dir: Directory for persistence
            collection_name: Collection name
            re_ranking_model: Re-ranking model (for future use)
            re_rank_top_k: Number of results to re-rank
            enable_reranking: Whether re-ranking is enabled
        """
        self.memory = memory
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._current_user: Optional[str] = None
        self._active_sessions: Dict[str, datetime] = {}

        # Re-ranking config (for future implementation)
        self.re_ranking_model = re_ranking_model
        self.re_rank_top_k = re_rank_top_k
        self.enable_reranking = enable_reranking

    # ==================== Session Management ====================

    def set_user_session(self, user_name: str) -> None:
        """
        Start a session for a specific user.

        Args:
            user_name: User identifier

        Example:
            memory.set_user_session("alice")
            memory.add("Alice's note")
        """
        if not user_name or not isinstance(user_name, str):
            raise ValueError("user_name must be a non-empty string")

        self._current_user = user_name
        self._active_sessions[user_name] = datetime.now()
        print(f"✓ Session started for user: {user_name}")

    def end_user_session(self, user_name: Optional[str] = None) -> None:
        """
        End a session for a specific user.

        Args:
            user_name: User identifier (defaults to current user)

        Example:
            memory.end_user_session("alice")
            # or
            memory.end_user_session()  # ends current user's session
        """
        target_user = user_name if user_name else self._current_user

        if not target_user:
            raise SessionNotSetError(
                "No active session to end. Use set_user_session() first."
            )

        if target_user in self._active_sessions:
            del self._active_sessions[target_user]

        if self._current_user == target_user:
            self._current_user = None

        print(f"✓ Session ended for user: {target_user}")

    def get_current_user(self) -> Optional[str]:
        """
        Get the current active user.

        Returns:
            Current user name or None
        """
        return self._current_user

    def get_active_sessions(self) -> Dict[str, str]:
        """
        Get all active sessions.

        Returns:
            Dictionary of user_name -> session_start_time (ISO format)
        """
        return {
            user: start_time.isoformat()
            for user, start_time in self._active_sessions.items()
        }

    def _ensure_session_active(self) -> None:
        """Ensure a user session is active before operations."""
        if not self._current_user:
            raise SessionNotSetError(
                "No active user session. Call set_user_session(user_name) before performing operations."
            )

    def _get_user_metadata(self, additional_metadata: Optional[Dict] = None) -> Dict:
        """
        Get metadata without user_id (since mem0 handles it separately).

        Args:
            additional_metadata: Additional metadata to merge

        Returns:
            Metadata dictionary
        """
        self._ensure_session_active()

        metadata = {}

        if additional_metadata:
            metadata.update(additional_metadata)

        return metadata

    # ==================== Core Memory Operations ====================

    def add(
            self,
            text: str,
            metadata: Optional[Dict] = None,
            user_id: Optional[str] = None
    ) -> Dict:
        """
        Add a memory with automatic user isolation.

        Args:
            text: Text content to store
            metadata: Additional metadata
            user_id: Override current user (for admin operations)

        Returns:
            Result from memory.add()

        Raises:
            SessionNotSetError: If no user session is active
        """
        if user_id:
            # Temporarily override user for this operation
            original_user = self._current_user
            self._current_user = user_id
            try:
                return self.add(text, metadata)
            finally:
                self._current_user = original_user

        self._ensure_session_active()

        # Prepare metadata (without user_id, as it's a separate parameter)
        full_metadata = metadata.copy() if metadata else {}
        full_metadata["timestamp"] = datetime.now().isoformat()

        # Pass user_id as a separate parameter to mem0
        result = self.memory.add(text, user_id=self._current_user, metadata=full_metadata)


        return result

    def query(
            self,
            query: str,
            limit: int = 5,
            filter:dict=None,
    ) -> List[Dict]:
        """
        Search memories with automatic user isolation.

        Args:
            query: Search query
            filter: add metadata to query
            limit: Maximum number of results

        Returns:
            List of search results

        Raises:
            SessionNotSetError: If no user session is active
        """

        self._ensure_session_active()

        # Pass user_id as parameter to mem0, not in filters
        try:
            results = []
            if filter is None:
                self.memory.search(query, user_id=self._current_user, limit=limit)
            else:
                self.memory.search(query, user_id=self._current_user,filters=filter, limit=limit)
            return results if results else []
        except Exception as e:
            print(f"Search error: {e}")
            return []






    # ==================== Utility Methods ====================

    def save(self) -> None:
        """
        Save memory to disk.
        Note: FAISS in mem0 auto-saves, so this is mostly a no-op.
        Kept for API compatibility.
        """
        try:
            # mem0's FAISS implementation auto-saves to the persist_dir
            # Check if there's a custom save method
            if hasattr(self.memory, 'vector_store'):
                vs = self.memory.vector_store

                # Try different save methods that might exist
                if hasattr(vs, 'save'):
                    vs.save()
                elif hasattr(vs, 'persist'):
                    vs.persist()
                elif hasattr(vs, '_save'):
                    vs._save()
                # If none exist, it auto-saves anyway
        except AttributeError:
            # FAISS auto-saves, so this is fine
            pass
        except Exception as e:
            print(f"Warning: Save operation encountered an issue: {e}")


    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a specific memory (with user session check).

        Args:
            memory_id: ID of memory to delete

        Returns:
            True if deleted successfully

        Raises:
            SessionNotSetError: If no user session is active
        """
        self._ensure_session_active()

        try:
            self.memory.delete(memory_id)
            self.save()
            return True
        except Exception as e:
            print(f"Error deleting memory: {e}")
            return False

    def clear_user_memories(self, user_id: Optional[str] = None) -> int:
        """
        Clear all memories for a specific user.

        Args:
            user_id: User to clear memories for (defaults to current user)

        Returns:
            Number of memories cleared

        Raises:
            SessionNotSetError: If no user session is active
        """
        target_user = user_id if user_id else self._current_user

        if not target_user:
            raise SessionNotSetError(
                "No active user session. Call set_user_session() first."
            )

        try:
            # Get all memories for the user
            all_memories = self.get_all(user_id=target_user)
            count = 0

            for mem in all_memories:
                if mem.get("id"):
                    if self.memory.delete(mem["id"]):
                        count += 1

            self.save()
            return count
        except Exception as e:
            print(f"Error clearing memories: {e}")
            return 0