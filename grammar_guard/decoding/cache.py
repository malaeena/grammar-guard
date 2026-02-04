"""
Cache management for compiled schemas and token indices.

Compiling schemas to FSMs and building token indices is expensive (5-30 seconds
depending on vocabulary size and schema complexity). We cache these to avoid
recomputation on subsequent uses.

Cache Structure:
    ~/.cache/grammar_guard/
    ├── indices/           # Token indices
    │   ├── {hash1}.pkl
    │   ├── {hash2}.pkl
    │   └── ...
    ├── fsms/             # Compiled FSMs
    │   ├── {hash1}.pkl
    │   └── ...
    └── metadata.json     # Cache metadata and stats

Cache Keys:
    - Hash of (schema, tokenizer_vocab_hash)
    - Ensures cache hit when same schema + tokenizer used
    - Different tokenizers = different cache entries

Usage:
    ```python
    from grammar_guard.decoding import get_or_build_index

    # Automatically uses cache
    token_index = get_or_build_index(
        schema={"type": "string"},
        tokenizer=tokenizer,
        force_rebuild=False
    )

    # First call: builds index (slow)
    # Subsequent calls: loads from cache (fast)
    ```
"""

import hashlib
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Manage cache for compiled schemas and token indices.

    Handles:
    - Cache directory management
    - Cache key computation
    - Cache hits/misses tracking
    - Cache cleanup (remove old entries)

    Attributes:
        cache_dir: Root cache directory
        enabled: Whether caching is enabled
        max_size_mb: Maximum cache size in megabytes
        max_age_days: Maximum age of cache entries in days
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        enabled: bool = True,
        max_size_mb: int = 1000,
        max_age_days: int = 30
    ):
        """
        Initialize cache manager.

        Args:
            cache_dir: Cache directory (default: ~/.cache/grammar_guard/)
            enabled: Whether to use caching
            max_size_mb: Maximum cache size
            max_age_days: Maximum age of entries before cleanup
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "grammar_guard"

        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        self.max_size_mb = max_size_mb
        self.max_age_days = max_age_days

        # Create cache directories
        if self.enabled:
            (self.cache_dir / "indices").mkdir(parents=True, exist_ok=True)
            (self.cache_dir / "fsms").mkdir(parents=True, exist_ok=True)

        # Load metadata
        self.metadata_path = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()

        logger.debug(f"CacheManager initialized at {self.cache_dir}")

    def get_index_path(self, cache_key: str) -> Path:
        """Get path for cached token index."""
        return self.cache_dir / "indices" / f"{cache_key}.pkl"

    def get_fsm_path(self, cache_key: str) -> Path:
        """Get path for cached FSM."""
        return self.cache_dir / "fsms" / f"{cache_key}.pkl"

    def has_index(self, cache_key: str) -> bool:
        """Check if index is cached."""
        if not self.enabled:
            return False
        return self.get_index_path(cache_key).exists()

    def has_fsm(self, cache_key: str) -> bool:
        """Check if FSM is cached."""
        if not self.enabled:
            return False
        return self.get_fsm_path(cache_key).exists()

    def load_index(self, cache_key: str) -> Optional[Any]:
        """
        Load cached token index.

        Args:
            cache_key: Cache key

        Returns:
            TokenIndex if found, None otherwise
        """
        if not self.enabled:
            return None

        path = self.get_index_path(cache_key)
        if not path.exists():
            logger.debug(f"Cache miss for index {cache_key[:8]}...")
            self._record_miss('index')
            return None

        try:
            with open(path, 'rb') as f:
                index = pickle.load(f)

            logger.info(f"Cache hit for index {cache_key[:8]}...")
            self._record_hit('index')
            return index

        except Exception as e:
            logger.warning(f"Failed to load cached index: {e}")
            self._record_miss('index')
            return None

    def save_index(self, cache_key: str, index: Any) -> Path:
        """
        Save token index to cache.

        Args:
            cache_key: Cache key
            index: TokenIndex to save

        Returns:
            Path where index was saved
        """
        if not self.enabled:
            return None

        path = self.get_index_path(cache_key)

        try:
            with open(path, 'wb') as f:
                pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info(f"Saved index to cache: {cache_key[:8]}...")
            self._record_save('index', path)
            return path

        except Exception as e:
            logger.error(f"Failed to save index to cache: {e}")
            return None

    def load_fsm(self, cache_key: str) -> Optional[Any]:
        """Load cached FSM."""
        if not self.enabled:
            return None

        path = self.get_fsm_path(cache_key)
        if not path.exists():
            logger.debug(f"Cache miss for FSM {cache_key[:8]}...")
            self._record_miss('fsm')
            return None

        try:
            with open(path, 'rb') as f:
                fsm = pickle.load(f)

            logger.info(f"Cache hit for FSM {cache_key[:8]}...")
            self._record_hit('fsm')
            return fsm

        except Exception as e:
            logger.warning(f"Failed to load cached FSM: {e}")
            self._record_miss('fsm')
            return None

    def save_fsm(self, cache_key: str, fsm: Any) -> Path:
        """Save FSM to cache."""
        if not self.enabled:
            return None

        path = self.get_fsm_path(cache_key)

        try:
            with open(path, 'wb') as f:
                pickle.dump(fsm, f, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info(f"Saved FSM to cache: {cache_key[:8]}...")
            self._record_save('fsm', path)
            return path

        except Exception as e:
            logger.error(f"Failed to save FSM to cache: {e}")
            return None

    def _load_metadata(self) -> Dict:
        """Load cache metadata."""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")

        # Default metadata
        return {
            'hits': {'index': 0, 'fsm': 0},
            'misses': {'index': 0, 'fsm': 0},
            'saves': {'index': 0, 'fsm': 0},
            'entries': {}
        }

    def _save_metadata(self) -> None:
        """Save cache metadata."""
        try:
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache metadata: {e}")

    def _record_hit(self, cache_type: str) -> None:
        """Record cache hit."""
        self.metadata['hits'][cache_type] = self.metadata['hits'].get(cache_type, 0) + 1
        self._save_metadata()

    def _record_miss(self, cache_type: str) -> None:
        """Record cache miss."""
        self.metadata['misses'][cache_type] = self.metadata['misses'].get(cache_type, 0) + 1
        self._save_metadata()

    def _record_save(self, cache_type: str, path: Path) -> None:
        """Record cache save."""
        self.metadata['saves'][cache_type] = self.metadata['saves'].get(cache_type, 0) + 1
        self.metadata['entries'][str(path)] = {
            'created': time.time(),
            'size': path.stat().st_size if path.exists() else 0
        }
        self._save_metadata()

    def get_stats(self) -> Dict:
        """
        Get cache statistics.

        Returns:
            Dict with cache stats

        Example:
            ```python
            stats = cache_manager.get_stats()
            print(f"Hit rate: {stats['hit_rate']:.1%}")
            print(f"Cache size: {stats['size_mb']:.1f} MB")
            ```
        """
        index_hits = self.metadata['hits'].get('index', 0)
        index_misses = self.metadata['misses'].get('index', 0)
        total_index = index_hits + index_misses

        fsm_hits = self.metadata['hits'].get('fsm', 0)
        fsm_misses = self.metadata['misses'].get('fsm', 0)
        total_fsm = fsm_hits + fsm_misses

        # Calculate cache size
        total_size = 0
        for entry in self.metadata.get('entries', {}).values():
            total_size += entry.get('size', 0)

        return {
            'enabled': self.enabled,
            'cache_dir': str(self.cache_dir),
            'index_hits': index_hits,
            'index_misses': index_misses,
            'index_hit_rate': index_hits / total_index if total_index > 0 else 0,
            'fsm_hits': fsm_hits,
            'fsm_misses': fsm_misses,
            'fsm_hit_rate': fsm_hits / total_fsm if total_fsm > 0 else 0,
            'num_entries': len(self.metadata.get('entries', {})),
            'size_bytes': total_size,
            'size_mb': total_size / (1024 * 1024)
        }

    def cleanup(self, dry_run: bool = False) -> Dict:
        """
        Clean up old cache entries.

        Removes entries older than max_age_days or if cache exceeds max_size_mb.

        Args:
            dry_run: If True, just report what would be deleted

        Returns:
            Dict with cleanup stats

        Example:
            ```python
            # See what would be cleaned
            stats = cache_manager.cleanup(dry_run=True)
            print(f"Would delete {stats['num_deleted']} entries")

            # Actually clean up
            stats = cache_manager.cleanup(dry_run=False)
            ```
        """
        if not self.enabled:
            return {'num_deleted': 0, 'bytes_freed': 0}

        now = time.time()
        max_age_seconds = self.max_age_days * 24 * 3600

        entries_to_delete = []
        bytes_to_free = 0

        for path_str, entry_info in self.metadata.get('entries', {}).items():
            path = Path(path_str)

            # Check age
            age = now - entry_info.get('created', 0)
            if age > max_age_seconds:
                entries_to_delete.append((path, entry_info))
                bytes_to_free += entry_info.get('size', 0)
                continue

        # If still over size limit, delete oldest entries
        current_size = sum(e.get('size', 0) for e in self.metadata.get('entries', {}).values())
        max_size_bytes = self.max_size_mb * 1024 * 1024

        if current_size > max_size_bytes:
            # Sort by age (oldest first)
            all_entries = [
                (Path(p), e) for p, e in self.metadata.get('entries', {}).items()
            ]
            all_entries.sort(key=lambda x: x[1].get('created', 0))

            # Delete until under limit
            for path, entry_info in all_entries:
                if current_size <= max_size_bytes:
                    break

                if (path, entry_info) not in entries_to_delete:
                    entries_to_delete.append((path, entry_info))
                    bytes_to_free += entry_info.get('size', 0)
                    current_size -= entry_info.get('size', 0)

        # Delete entries
        num_deleted = 0
        if not dry_run:
            for path, entry_info in entries_to_delete:
                try:
                    if path.exists():
                        path.unlink()
                    del self.metadata['entries'][str(path)]
                    num_deleted += 1
                except Exception as e:
                    logger.warning(f"Failed to delete {path}: {e}")

            self._save_metadata()

        return {
            'num_deleted': num_deleted if not dry_run else len(entries_to_delete),
            'bytes_freed': bytes_to_free,
            'mb_freed': bytes_to_free / (1024 * 1024)
        }

    def clear(self) -> None:
        """
        Clear entire cache.

        WARNING: This deletes all cached indices and FSMs.

        Example:
            ```python
            cache_manager.clear()  # Delete everything
            ```
        """
        if not self.enabled:
            return

        import shutil

        try:
            shutil.rmtree(self.cache_dir)
            logger.info(f"Cleared cache directory: {self.cache_dir}")

            # Recreate directories
            (self.cache_dir / "indices").mkdir(parents=True, exist_ok=True)
            (self.cache_dir / "fsms").mkdir(parents=True, exist_ok=True)

            # Reset metadata
            self.metadata = {
                'hits': {'index': 0, 'fsm': 0},
                'misses': {'index': 0, 'fsm': 0},
                'saves': {'index': 0, 'fsm': 0},
                'entries': {}
            }
            self._save_metadata()

        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")


# Global cache manager instance
_global_cache_manager = None


def get_cache_manager() -> CacheManager:
    """
    Get global cache manager instance.

    Returns:
        CacheManager: Shared cache manager

    Example:
        ```python
        from grammar_guard.decoding import get_cache_manager

        cache = get_cache_manager()
        stats = cache.get_stats()
        ```
    """
    global _global_cache_manager

    if _global_cache_manager is None:
        _global_cache_manager = CacheManager()

    return _global_cache_manager


def compute_schema_hash(schema: Dict) -> str:
    """
    Compute stable hash of a JSON Schema.

    Args:
        schema: JSON Schema dictionary

    Returns:
        str: Hex hash string

    Example:
        ```python
        hash1 = compute_schema_hash({"type": "string"})
        hash2 = compute_schema_hash({"type": "string"})
        assert hash1 == hash2  # Same schema = same hash
        ```
    """
    # Serialize to stable JSON (sorted keys)
    schema_json = json.dumps(schema, sort_keys=True)

    # Hash
    return hashlib.sha256(schema_json.encode()).hexdigest()


def compute_cache_key(schema: Dict, tokenizer_name: str) -> str:
    """
    Compute cache key for (schema, tokenizer) pair.

    Args:
        schema: JSON Schema dictionary
        tokenizer_name: Tokenizer name/path

    Returns:
        str: Cache key (hex hash)

    Example:
        ```python
        key = compute_cache_key(
            schema={"type": "string"},
            tokenizer_name="gpt2"
        )
        ```
    """
    schema_hash = compute_schema_hash(schema)
    combined = f"{schema_hash}:{tokenizer_name}"
    return hashlib.sha256(combined.encode()).hexdigest()
